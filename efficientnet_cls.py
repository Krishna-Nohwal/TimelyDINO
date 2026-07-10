"""
EfficientNet CLS-style Stage 1 training script.

This is the non-spherical EfficientNet counterpart to spherical_efficientnet.py.
It keeps the same FF++ real/fake manifest split, CDFv1 evaluation, image
pipeline, metrics, optimizer schedule, and checkpointing, but uses a standard
learnable linear classifier:

  image -> EfficientNet pooled features -> linear classifier
  loss  = CrossEntropy(logits, labels) + lambda * SupCon(features, labels)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.optim as optim
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentations import augment_batch, load_and_resize, normalize


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Stage 1 EfficientNet training: pooled features + linear classifier"
)
parser.add_argument("--epochs",        default=50,   type=int)
parser.add_argument("--batch_size",    default=64,   type=int)
parser.add_argument("--num_workers",   default=8,    type=int)
parser.add_argument("--save_root",     default="checkpoints_efficientnet_cls", type=str)
parser.add_argument("--load_from",     default="",   type=str)
parser.add_argument("--manifest",      default="E:/Work/sampled_30k/manifest_onct.csv", type=str,
                    help="FF++ training manifest CSV with sample_dir,label.")
parser.add_argument("--root_dir",      default="E:/Work/sampled_30k/", type=str,
                    help="FF++ frame root used with --manifest.")
parser.add_argument("--cdf_root",      default="E:/Work/cdfv1_onct_out", type=str,
                    help="CDFv1 frame root used only for testing.")
parser.add_argument("--cdf_csv",       default="E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str,
                    help="CDFv1 manifest CSV used only for testing.")
parser.add_argument("--model_name",    default="tf_efficientnet_b4_ns", type=str)
parser.add_argument("--image_size",    default=256,  type=int)
parser.add_argument("--val_ratio",     default=0.05, type=float)
parser.add_argument("--lr",            default=1e-4, type=float)
parser.add_argument("--weight_decay",  default=1e-4, type=float)
parser.add_argument("--warmup_steps",  default=512,  type=int)
parser.add_argument("--supcon_weight", default=1/16, type=float)
parser.add_argument("--max_train_samples", default=0, type=int,
                    help="Optional cap on training images. 0 uses all training images.")
parser.add_argument("--no_amp",        action="store_true")
parser.add_argument("--no_compile",    action="store_true")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = device.type == "cuda" and not args.no_amp
_num_workers = args.num_workers

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")
print(f"AMP enabled : {amp_enabled}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EfficientNetCLS(nn.Module):
    """timm EfficientNet with pooled features and a linear real/fake head."""

    NUM_CLASSES = 2

    def __init__(self, model_name: str = "tf_efficientnet_b4_ns"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.classifier = nn.Linear(int(self.backbone.num_features), self.NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features.float())
        return logits, features


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def limit_split_by_label(df: pd.DataFrame, max_samples: int, split_name: str) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df

    labels = sorted(df["label"].unique().tolist())
    base = max_samples // len(labels)
    allocations = {
        label: min(int((df["label"] == label).sum()), base)
        for label in labels
    }

    remaining = max_samples - sum(allocations.values())
    while remaining > 0:
        candidates = [
            label for label in labels
            if allocations[label] < int((df["label"] == label).sum())
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda label: int((df["label"] == label).sum()) - allocations[label],
            reverse=True,
        )
        allocations[candidates[0]] += 1
        remaining -= 1

    limited_parts = [
        df[df["label"] == label].sample(n=allocations[label], random_state=42)
        for label in labels
        if allocations[label] > 0
    ]
    limited_df = pd.concat(limited_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    counts = limited_df["label"].value_counts().sort_index().to_dict()
    print(f"{split_name} capped to {len(limited_df)} samples -> {counts}")
    return limited_df


def prepare_splits(
    manifest_csv: str,
    root_dir: str,
    val_ratio: float = 0.05,
    max_train_samples: int = 0,
):
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    real_pool = df[df["label"] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    fake_pool = df[df["label"] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"FF++ full dataset -> Real: {len(real_pool)} | Fake: {len(fake_pool)}")

    real_val_n = int(len(real_pool) * val_ratio)
    fake_val_n = int(len(fake_pool) * val_ratio)

    real_val = real_pool.iloc[:real_val_n]
    real_train = real_pool.iloc[real_val_n:]
    fake_val = fake_pool.iloc[:fake_val_n]
    fake_train = fake_pool.iloc[fake_val_n:]

    train_df = pd.concat([real_train, fake_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = pd.concat([real_val, fake_val]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df = limit_split_by_label(train_df, max_train_samples, "Train")
    train_real_n = int((train_df["label"] == 0).sum())
    train_fake_n = int((train_df["label"] == 1).sum())

    print(f"FF++ Train -> Real: {train_real_n} | Fake: {train_fake_n} | Total: {len(train_df)}")
    print(f"FF++ Val   -> Real: {len(real_val)} | Fake: {len(fake_val)} | Total: {len(val_df)}")
    return train_df, val_df


def image_path_from_manifest_sample(sample_dir: str, root_dir: str) -> str:
    sample_dir = str(sample_dir).replace("\\", "/")
    root = Path(root_dir)

    if "sampled_30k/" in sample_dir:
        sample_dir = sample_dir.split("sampled_30k/", 1)[-1]

    sample_path = Path(sample_dir)
    if sample_path.is_absolute():
        return str(sample_path / "image.png")
    return str(root / sample_dir / "image.png")


class ManifestImageDataset(Dataset):
    """Train/val dataset. label: 0=Real, 1=Fake."""

    def __init__(self, df: pd.DataFrame, root_dir: str, image_size: int):
        paths = df["sample_dir"].apply(lambda sample: image_path_from_manifest_sample(sample, root_dir))
        labels = df["label"].astype(int).values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [Dataset] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))
        self.image_size = image_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, self.image_size)
        img = normalize(img)
        return img, label


class CDFv1Dataset(Dataset):
    """CDFv1 test dataset."""

    def __init__(self, csv_path: str, data_root: str, image_size: int):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = df["label"].astype(int)

        print(f"CDFv1 -> Real: {(df['label'] == 0).sum()} | Fake: {(df['label'] == 1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        paths = df["sample_dir"].apply(lambda d: str(root / str(d) / "image.png"))
        labels = df["label"].values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))
        self.image_size = image_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, self.image_size)
        img = normalize(img)
        return img, label


# ---------------------------------------------------------------------------
# Loss / Metrics
# ---------------------------------------------------------------------------

ce_loss = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()


def efficientnet_cls_loss(logits, features, labels, supcon_weight):
    return ce_loss(logits, labels) + supcon_weight * supcon_loss(features, labels)


def compute_metrics(all_labels, all_probs, split_name: str, epoch: int):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  [{split_name}] Epoch {epoch+1:02d} | "
          f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc*100:.2f}%  F1={f1:.4f}  EER={eer*100:.2f}%  "
          f"TPR={tpr*100:.2f}%  FPR={fpr*100:.2f}%  TNR={tnr*100:.2f}%  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")

    return auc


def autocast_context():
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)


def run_eval(model, loader, desc):
    all_labels, all_probs = [], []
    model.eval()
    with torch.inference_mode(), autocast_context():
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits, _ = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_probs


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 80

    train_df, val_df = prepare_splits(
        args.manifest,
        args.root_dir,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
    )
    train_dataset = ManifestImageDataset(train_df, args.root_dir, args.image_size)
    val_dataset = ManifestImageDataset(val_df, args.root_dir, args.image_size)
    cdf_dataset = CDFv1Dataset(args.cdf_csv, args.cdf_root, args.image_size)

    _persistent = _num_workers > 0
    _prefetch = 4 if _num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=_num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=_num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    cdf_loader = DataLoader(
        cdf_dataset,
        batch_size=args.batch_size,
        num_workers=_num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )

    os.makedirs(args.save_root, exist_ok=True)

    model = EfficientNetCLS(model_name=args.model_name).to(device)
    print(f"Model       : {args.model_name}")
    print(f"Image size  : {args.image_size}")
    print(f"Feature dim : {model.backbone.num_features}")

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location="cpu"))
        print(f"Loaded checkpoint from {args.load_from}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    scaler = make_grad_scaler()

    lr_base = args.lr
    epochs = args.epochs
    iter_per_epoch = len(train_loader)
    total_steps = epochs * iter_per_epoch
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 1))
    lr_min = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmup_steps) * math.pi / max(total_steps - warmup_steps, 1))) / 2) + lr_min)
            if i > warmup_steps
            else (i / max(warmup_steps, 1) + lr_min)
        )
        for i in range(total_steps)
    }

    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[min(step, total_steps - 1)]
    )

    best_test_auc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)

        model.train()
        iter_i = epoch * iter_per_epoch
        train_labels, train_probs = [], []

        for batch_idx, (imgs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        ):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs = augment_batch(imgs)

            with autocast_context():
                logits, features = model(imgs)
                loss = efficientnet_cls_loss(logits, features, labels, args.supcon_weight)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 100 == 0:
                print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

        print()
        compute_metrics(train_labels, train_probs, "Train", epoch)

        val_labels, val_probs = run_eval(model, val_loader, f"Epoch {epoch+1} [val]")
        compute_metrics(val_labels, val_probs, "Val  ", epoch)

        cdf_labels, cdf_probs = run_eval(model, cdf_loader, f"Epoch {epoch+1} [CDFv1]")
        test_auc = compute_metrics(cdf_labels, cdf_probs, "Test ", epoch)

        live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = live_module.state_dict()

        torch.save(state_dict, os.path.join(args.save_root, "latest.pth"))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            print(f"\n  New best Test AUC={best_test_auc:.4f} -> saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Training complete. Best checkpoint: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
