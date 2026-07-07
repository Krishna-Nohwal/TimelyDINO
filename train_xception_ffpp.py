"""
Train a pretrained Xception frame classifier on FaceForensics++ frames.

Expected FF++ layout:
    <root_dir>/<sample_dir>/image.png

Expected manifest columns:
    sample_dir,label

Labels are assumed to be:
    0 = real, 1 = fake

Example:
python train_xception_ffpp.py \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out \
    --save_dir checkpoints_xception_ffpp \
    --epochs 10 \
    --batch_size 64
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Train pretrained Xception on FF++ frames.")
    p.add_argument("--manifest", required=True, help="FF++ manifest CSV with sample_dir,label.")
    p.add_argument("--root_dir", required=True, help="FF++ preprocessed frame root.")
    p.add_argument("--save_dir", default="checkpoints_xception_ffpp")
    p.add_argument("--model_name", default="xception", help="timm model name. Try 'legacy_xception' if needed.")
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--max_val_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def video_id_from_sample_dir(sample_dir: str) -> str:
    sample_dir = str(sample_dir).replace("\\", "/")
    parts = Path(sample_dir).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = basename.rfind("_frame_")
    if idx != -1:
        clip_id = basename[:idx]
    else:
        match = re.search(r"_f\d+$", basename)
        clip_id = basename[:match.start()] if match else basename
    return f"{prefix}/{clip_id}" if prefix else clip_id


def prepare_video_split(manifest: str, root_dir: str, val_ratio: float, seed: int):
    df = pd.read_csv(manifest)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)

    rng = np.random.default_rng(seed)
    real_vids = rng.permutation(df[df["label"] == 0]["video_id"].unique())
    fake_vids = rng.permutation(df[df["label"] == 1]["video_id"].unique())

    real_val_n = max(1, int(len(real_vids) * val_ratio))
    fake_val_n = max(1, int(len(fake_vids) * val_ratio))
    val_ids = set(real_vids[:real_val_n]) | set(fake_vids[:fake_val_n])

    train_df = df[~df["video_id"].isin(val_ids)].reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_ids)].reset_index(drop=True)

    root = Path(root_dir)
    train_df["path"] = train_df["sample_dir"].astype(str).str.replace("\\", "/", regex=False).apply(
        lambda rel: str(root / rel / "image.png")
    )
    val_df["path"] = val_df["sample_dir"].astype(str).str.replace("\\", "/", regex=False).apply(
        lambda rel: str(root / rel / "image.png")
    )

    train_df = train_df[train_df["path"].apply(os.path.exists)].reset_index(drop=True)
    val_df = val_df[val_df["path"].apply(os.path.exists)].reset_index(drop=True)

    print("FF++ split:")
    print(f"  train frames: {len(train_df)}  real={(train_df['label'] == 0).sum()}  fake={(train_df['label'] == 1).sum()}")
    print(f"  val frames  : {len(val_df)}  real={(val_df['label'] == 0).sum()}  fake={(val_df['label'] == 1).sum()}")
    print(f"  train videos: {train_df['video_id'].nunique()}  val videos: {val_df['video_id'].nunique()}")
    return train_df, val_df


def cap_df(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=seed).reset_index(drop=True)


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, train: bool):
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.image_size = image_size
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = self.tf(img)
        except Exception:
            img = torch.zeros(3, self.image_size, self.image_size)
        return img, label


def build_model(model_name: str):
    import timm

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=2)
    except Exception as exc:
        if model_name == "xception":
            print("Could not create timm model 'xception'. Retrying 'legacy_xception'.")
            return timm.create_model("legacy_xception", pretrained=True, num_classes=2)
        raise exc


def make_class_weight(train_df: pd.DataFrame, device: torch.device):
    counts = train_df["label"].value_counts().to_dict()
    real = max(int(counts.get(0, 0)), 1)
    fake = max(int(counts.get(1, 0)), 1)
    total = real + fake
    weights = torch.tensor([total / (2 * real), total / (2 * fake)], dtype=torch.float32)
    print(f"Class weights: real={weights[0]:.4f}, fake={weights[1]:.4f}")
    return weights.to(device)


@torch.inference_mode()
def evaluate(model, loader, device, amp_enabled: bool, name: str):
    model.eval()
    labels_all, probs_all = [], []
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)
    with autocast:
        for images, labels in tqdm(loader, desc=name, leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            labels_all.extend(labels.numpy().astype(int).tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    probs_np = np.asarray(probs_all, dtype=np.float64)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    auc = roc_auc_score(labels_np, probs_np)
    ap = average_precision_score(labels_np, probs_np)
    acc = accuracy_score(labels_np, preds_np)
    print(f"  [{name}] AUC={auc:.4f}  AP={ap:.4f}  Acc={acc * 100:.2f}%")
    return auc, ap, acc


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp_enabled: bool, epoch: int):
    model.train()
    total_loss = 0.0
    total_seen = 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)
    pbar = tqdm(loader, desc=f"Epoch {epoch} train")
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_seen += bs
        if step % 20 == 0:
            pbar.set_postfix(loss=total_loss / max(total_seen, 1))
    return total_loss / max(total_seen, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Train pretrained Xception on FF++")
    print("=" * 88)
    print(f"Device      : {device}")
    print(f"AMP enabled : {amp_enabled}")
    print(f"Model       : {args.model_name}")
    print(f"Manifest    : {args.manifest}")
    print(f"Root        : {args.root_dir}")
    print(f"Save dir    : {save_dir}")

    train_df, val_df = prepare_video_split(args.manifest, args.root_dir, args.val_ratio, args.seed)
    train_df = cap_df(train_df, args.max_train_frames, args.seed)
    val_df = cap_df(val_df, args.max_val_frames, args.seed + 1)

    train_set = FrameDataset(train_df, args.image_size, train=True)
    val_set = FrameDataset(val_df, args.image_size, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=make_class_weight(train_df, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_auc = -math.inf
    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 88)
        print(f"Epoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.3e}")
        print("=" * 88)
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, amp_enabled, epoch)
        print(f"  mean train loss: {loss:.4f}")

        val_auc, val_ap, val_acc = evaluate(model, val_loader, device, amp_enabled, "FF++ Val")
        scheduler.step()

        state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auc": val_auc,
            "val_ap": val_ap,
            "val_acc": val_acc,
            "args": vars(args),
        }
        torch.save(state, save_dir / "last.pth")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(state, save_dir / "best.pth")
            print(f"  saved new best: {save_dir / 'best.pth'}  val_auc={best_auc:.4f}")

    print("\nDone.")
    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
