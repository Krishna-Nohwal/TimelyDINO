"""
Train a pretrained Xception frame classifier on FF++ plus extra datasets.

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
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --cdfv3_root /media/tarun/B482367C823642E2/usr/cdfv3_face_crops \
    --dfd_fake_root /media/tarun/B482367C823642E2/usr/dfd_faces/fake \
    --dfd_real_root /media/tarun/B482367C823642E2/usr/dfd_faces/real \
    --df0_fake_root /media/tarun/B482367C823642E2/usr/df1.0_faces/fake \
    --df0_real_root /media/tarun/B482367C823642E2/usr/df1.0_faces/real \
    --dfdc_fake_root /media/tarun/B482367C823642E2/usr/dfdc/fake \
    --dfdc_real_root /media/tarun/B482367C823642E2/usr/dfdc/real \
    --wdf_fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --wdf_real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --uadfv_fake_root /media/tarun/B482367C823642E2/usr/uadfv_faces/fake \
    --uadfv_real_root /media/tarun/B482367C823642E2/usr/uadfv_faces/real \
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
    p = argparse.ArgumentParser(description="Train pretrained Xception on FF++ plus optional extra datasets.")
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
    p.add_argument("--frames_per_video", type=int, default=0, help="0 = use all frames from every selected video.")
    p.add_argument("--max_train_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--max_val_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--no_amp", action="store_true")

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # CDFv3 / CDF++ manifest layout. Manifest labels: 1=Real, 0=Fake.
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # UADFV / DFo nested layouts: <root>/<video>/<frame>/image.png
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC / WDF flat layouts: <root>/<video_id>_<frame>.png
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")
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


def sample_frame_paths(paths: list[str], frames_per_video: int) -> list[str]:
    if frames_per_video <= 0 or len(paths) <= frames_per_video:
        return paths
    indices = np.linspace(0, len(paths) - 1, frames_per_video, dtype=int)
    return [paths[int(i)] for i in indices]


def print_video_counts(name: str, videos: list[tuple[str, list[str], int]]):
    real_n = sum(1 for _, _, label in videos if label == 0)
    fake_n = sum(1 for _, _, label in videos if label == 1)
    frame_n = sum(len(paths) for _, paths, _ in videos)
    print(f"  [{name}] videos={len(videos)}  frames={frame_n}  real_videos={real_n}  fake_videos={fake_n}")


def build_ffpp_videos(manifest: str, root_dir: str):
    df = pd.read_csv(manifest)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)
    root = Path(root_dir)

    videos = []
    for video_id, group in df.groupby("video_id"):
        label = int(group["label"].iloc[0])
        paths = []
        for rel in group["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
            path = root / rel / "image.png"
            if path.is_file():
                paths.append(str(path))
        if paths:
            videos.append((str(video_id), sorted(paths), label))
    print_video_counts("FFPP", videos)
    return videos


def video_id_from_sample(sample_name: str) -> str:
    return re.sub(r"_(?:frame_|f)\d+$", "", sample_name)


def build_cdfv2_videos(fake_root: str, real_root: str):
    vid2paths, vid2label = {}, {}
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [CDFv2] WARNING: missing root {root}")
            continue
        for d in sorted(root.iterdir()):
            path = d / "image.png"
            if d.is_dir() and path.exists():
                vid = video_id_from_sample(d.name)
                vid2paths.setdefault(vid, []).append(str(path))
                vid2label[vid] = label
    videos = [(vid, sorted(paths), vid2label[vid]) for vid, paths in sorted(vid2paths.items())]
    print_video_counts("CDFv2", videos)
    return videos


def video_id_from_cdfv3_sample_dir(sample_dir: str) -> str:
    return Path(str(sample_dir).replace("\\", "/")).parent.name


def build_cdfv3_videos(cdfv3_csv: str, cdfv3_root: str):
    df = pd.read_csv(cdfv3_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CDFv3/CDF++ manifest must contain {required}. Found: {list(df.columns)}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(video_id_from_cdfv3_sample_dir)
    root = Path(cdfv3_root)

    videos = []
    for video_id, group in df.groupby("video_id"):
        manifest_label = int(group["label"].iloc[0])
        label = 0 if manifest_label == 1 else 1
        paths = []
        for rel in group["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
            path = root / rel / "image.png"
            if path.is_file():
                paths.append(str(path))
        if paths:
            videos.append((str(video_id), sorted(paths), label))
    print_video_counts("CDFv3", videos)
    return videos


def sort_nested_frame_paths(paths: list[Path]) -> list[str]:
    def key_fn(path: Path):
        parent = path.parent.name
        if parent.isdigit():
            return int(parent), str(path)
        return 10**12, str(path)
    return [str(path) for path in sorted(paths, key=key_fn)]


def build_nested_image_videos(fake_root: str, real_root: str, dataset_name: str):
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        for video_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            paths = sort_nested_frame_paths(list(video_dir.rglob("image.png")))
            if paths:
                videos.append((video_dir.name, paths, label))
    print_video_counts(dataset_name, videos)
    return videos


FLAT_FRAME_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def build_flat_videos(fake_root: str, real_root: str, dataset_name: str):
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        grouped = {}
        skipped = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = FLAT_FRAME_RE.match(path.name)
            if not match:
                skipped += 1
                continue
            video_id, frame_idx = match.group(1), int(match.group(2))
            grouped.setdefault(video_id, []).append((frame_idx, str(path)))
        if skipped:
            print(f"  [{dataset_name}] skipped {skipped} files under {root}")
        for video_id, indexed_paths in sorted(grouped.items()):
            paths = [p for _, p in sorted(indexed_paths)]
            if paths:
                videos.append((video_id, paths, label))
    print_video_counts(dataset_name, videos)
    return videos


def split_videos(videos: list[tuple[str, list[str], int]], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(set(label for _, _, label in videos)):
        class_videos = [v for v in videos if v[2] == label]
        if not class_videos:
            continue
        order = rng.permutation(len(class_videos))
        class_videos = [class_videos[int(i)] for i in order]
        val_n = max(1, int(len(class_videos) * val_ratio)) if val_ratio > 0 and len(class_videos) > 1 else 0
        val.extend(class_videos[:val_n])
        train.extend(class_videos[val_n:])
    return train, val


def videos_to_df(videos: list[tuple[str, list[str], int]], dataset_name: str, frames_per_video: int):
    rows = []
    for video_id, paths, label in videos:
        for path in sample_frame_paths(paths, frames_per_video):
            rows.append({
                "path": path,
                "label": int(label),
                "dataset": dataset_name,
                "video_id": video_id,
            })
    return pd.DataFrame(rows)


def build_all_videos(args):
    dataset_videos = {}
    print("\nBuilding dataset video lists ...")
    dataset_videos["FFPP"] = build_ffpp_videos(args.manifest, args.root_dir)

    if args.cdfv2_fake_root and args.cdfv2_real_root:
        dataset_videos["CDFv2"] = build_cdfv2_videos(args.cdfv2_fake_root, args.cdfv2_real_root)
    else:
        print("  [skip] CDFv2 roots not provided.")

    if args.cdfv3_root:
        cdfv3_csv = args.cdfv3_csv or str(Path(args.cdfv3_root) / "manifest_cdfv3_face_crops.csv")
        dataset_videos["CDFv3"] = build_cdfv3_videos(cdfv3_csv, args.cdfv3_root)
    else:
        print("  [skip] CDFv3/CDF++ root not provided.")

    dfo_fake = args.dfo_fake_root or args.df0_fake_root
    dfo_real = args.dfo_real_root or args.df0_real_root
    if dfo_fake and dfo_real:
        dataset_videos["DFo"] = build_nested_image_videos(dfo_fake, dfo_real, "DFo")
    else:
        print("  [skip] DFo roots not provided.")

    if args.dfd_fake_root and args.dfd_real_root:
        dataset_videos["DFD"] = build_nested_image_videos(args.dfd_fake_root, args.dfd_real_root, "DFD")
    else:
        print("  [skip] DFD roots not provided.")

    if args.uadfv_fake_root and args.uadfv_real_root:
        dataset_videos["UADFV"] = build_nested_image_videos(args.uadfv_fake_root, args.uadfv_real_root, "UADFV")
    else:
        print("  [skip] UADFV roots not provided.")

    if args.dfdc_fake_root and args.dfdc_real_root:
        dataset_videos["DFDC"] = build_flat_videos(args.dfdc_fake_root, args.dfdc_real_root, "DFDC")
    else:
        print("  [skip] DFDC roots not provided.")

    if args.wdf_fake_root and args.wdf_real_root:
        dataset_videos["WDF"] = build_flat_videos(args.wdf_fake_root, args.wdf_real_root, "WDF")
    else:
        print("  [skip] WDF roots not provided.")

    return {name: videos for name, videos in dataset_videos.items() if videos}


def prepare_video_split(args):
    dataset_videos = build_all_videos(args)
    train_dfs, val_dfs = [], []
    print("\nVideo-level train/val splits:")
    for offset, (dataset_name, videos) in enumerate(sorted(dataset_videos.items()), start=100):
        train_videos, val_videos = split_videos(videos, args.val_ratio, args.seed + offset)
        train_df = videos_to_df(train_videos, dataset_name, args.frames_per_video)
        val_df = videos_to_df(val_videos, dataset_name, args.frames_per_video)
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        print(
            f"  {dataset_name:<6} train_videos={len(train_videos):>5} val_videos={len(val_videos):>4} "
            f"train_frames={len(train_df):>7} val_frames={len(val_df):>6}"
        )

    if not train_dfs or not val_dfs:
        raise ValueError("No train/val data found.")

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    print("\nCombined split:")
    print(f"  train frames: {len(train_df)}  real={(train_df['label'] == 0).sum()}  fake={(train_df['label'] == 1).sum()}")
    print(f"  val frames  : {len(val_df)}  real={(val_df['label'] == 0).sum()}  fake={(val_df['label'] == 1).sum()}")
    return train_df, val_df


def cap_df(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df
    reals = df[df["label"] == 0]
    fakes = df[df["label"] == 1]
    if reals.empty or fakes.empty:
        return df.sample(n=cap, random_state=seed).reset_index(drop=True)
    n_real = min(len(reals), cap // 2)
    n_fake = min(len(fakes), cap - n_real)
    n_real = min(len(reals), cap - n_fake)
    capped = pd.concat([
        reals.sample(n=n_real, random_state=seed),
        fakes.sample(n=n_fake, random_state=seed + 1),
    ])
    return capped.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


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

    acc = accuracy_score(labels_np, preds_np)
    if len(np.unique(labels_np)) < 2:
        auc = float("nan")
        ap = float("nan")
        print(f"  [{name}] AUC=nan  AP=nan  Acc={acc * 100:.2f}%  (single class)")
        return auc, ap, acc
    auc = roc_auc_score(labels_np, probs_np)
    ap = average_precision_score(labels_np, probs_np)
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
    print("Train pretrained Xception on FF++ + extra datasets")
    print("=" * 88)
    print(f"Device      : {device}")
    print(f"AMP enabled : {amp_enabled}")
    print(f"Model       : {args.model_name}")
    print(f"Manifest    : {args.manifest}")
    print(f"Root        : {args.root_dir}")
    print(f"Save dir    : {save_dir}")

    train_df, val_df = prepare_video_split(args)
    train_df = cap_df(train_df, args.max_train_frames, args.seed)
    val_df = cap_df(val_df, args.max_val_frames, args.seed + 1)
    print("\nAfter optional frame caps:")
    print(f"  train frames: {len(train_df)}  real={(train_df['label'] == 0).sum()}  fake={(train_df['label'] == 1).sum()}")
    print(f"  val frames  : {len(val_df)}  real={(val_df['label'] == 0).sum()}  fake={(val_df['label'] == 1).sum()}")

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

        val_auc, val_ap, val_acc = evaluate(model, val_loader, device, amp_enabled, "Combined Val")
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
