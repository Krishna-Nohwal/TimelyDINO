"""
Patch-attention Stage 1 training script.

This trains a DINOv3 ViT-L/16 model using only final-layer patch tokens:

  patch_feat = AttentionPool(final_layer_patch_tokens)
  logits     = classifier(patch_feat)
  loss       = CrossEntropy(logits, labels) + lambda * SupCon(patch_feat, labels)

There are no CLS-token predictions, no reg-token features, no SpatialHeads, no
auxiliary tapped layers, and no MultiSimilarity loss.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
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
    description="Stage 1 patch-attention training: final-layer patch tokens + CE/SupCon"
)
parser.add_argument("--epochs",        default=50,   type=int)
parser.add_argument("--batch_size",    default=128,  type=int)
parser.add_argument("--num_workers",   default=36,   type=int)
parser.add_argument("--save_root",     default="checkpoints_patch_attn", type=str)
parser.add_argument("--load_from",     default="",   type=str)
parser.add_argument("--manifest",      default="E:/Work/sampled_30k/manifest_onct.csv", type=str)
parser.add_argument("--root_dir",      default="E:/Work/sampled_30k/", type=str)
parser.add_argument("--cdf_root",      default="E:/Work/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv",       default="E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
parser.add_argument("--val_ratio",     default=0.05, type=float)
parser.add_argument("--lr",            default=1e-4, type=float)
parser.add_argument("--warmup_steps",  default=512,  type=int)
parser.add_argument("--supcon_weight", default=1/16, type=float)
parser.add_argument("--no_compile",    action="store_true")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Multi-head learnable attention pooling over patch tokens.

    This mirrors the pooling style in frame_model.py: raw patch tokens act as
    keys/values, each head has a learned query, and a learned scalar positional
    bias is added to attention logits before softmax.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 4, num_patches: int = 256):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_patches = num_patches
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.pos_bias = nn.Parameter(torch.zeros(1, 1, 1, num_patches))

    def _positional_bias(self, n: int, device, dtype):
        bias = self.pos_bias
        if n != self.num_patches:
            bias = nn.functional.interpolate(
                bias.reshape(1, 1, self.num_patches),
                size=n,
                mode="linear",
                align_corners=False,
            ).reshape(1, 1, 1, n)
        return bias.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.query.expand(B, -1, -1, -1)

        attn = (q @ x_heads.transpose(-2, -1)) * self.scale
        attn = attn + self._positional_bias(N, x.device, attn.dtype)
        attn = attn.softmax(dim=-1)

        out = attn @ x_heads
        return out.permute(0, 2, 1, 3).reshape(B, C)


class PatchAttnViT(nn.Module):
    """DINOv3 ViT-L/16 using attention-pooled final-layer patch tokens."""

    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_PATCHES = 256
    NUM_POOL_HEADS = 4

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["attn.qkv"],
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)
        self.patch_pool = AttentionPool(
            self.EMBED_DIM,
            num_heads=self.NUM_POOL_HEADS,
            num_patches=self.NUM_PATCHES,
        )
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        spatial_map, _ = intermediates[0]
        B, C, H, W = spatial_map.shape
        patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        patch_feat = self.patch_pool(patch_tokens)
        logits = self.classifier(patch_feat.float())
        return logits, patch_feat


def check_layerscale(vit_backbone):
    try:
        block0 = vit_backbone.blocks[0]
        has_ls = hasattr(block0, "ls1") and not isinstance(block0.ls1, nn.Identity)
        print(f"  LayerScale present in backbone blocks: {has_ls}")
        return has_ls
    except Exception as exc:
        print(f"  Could not verify LayerScale presence: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    real_pool = df[df["label"] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    fake_pool = df[df["label"] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Full dataset -> Real: {len(real_pool)} | Fake: {len(fake_pool)}")

    real_val_n = int(len(real_pool) * val_ratio)
    fake_val_n = int(len(fake_pool) * val_ratio)

    real_val = real_pool.iloc[:real_val_n]
    real_train = real_pool.iloc[real_val_n:]
    fake_val = fake_pool.iloc[:fake_val_n]
    fake_train = fake_pool.iloc[fake_val_n:]

    train_df = pd.concat([real_train, fake_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = pd.concat([real_val, fake_val]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Train -> Real: {len(real_train)} | Fake: {len(fake_train)} | Total: {len(train_df)}")
    print(f"Val   -> Real: {len(real_val)} | Fake: {len(fake_val)} | Total: {len(val_df)}")
    return train_df, val_df


class ManifestImageDataset(Dataset):
    """Train/val dataset. label: 0=Real, 1=Fake."""

    def __init__(self, df: pd.DataFrame, root_dir: str):
        paths = (
            df["sample_dir"]
            .str.replace("\\", "/", regex=False)
            .str.split("sampled_30k/", n=1)
            .str[-1]
            .apply(lambda rel: os.path.join(root_dir, rel, "image.png"))
        )
        labels = df["label"].astype(int).values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [Dataset] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


class CDFv1Dataset(Dataset):
    """CDFv1 test dataset."""

    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = df["label"].astype(int)

        print(f"CDFv1 -> Real: {(df['label'] == 0).sum()} | Fake: {(df['label'] == 1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        paths = df["sample_dir"].apply(lambda d: str(root / d / "image.png"))
        labels = df["label"].values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


# ---------------------------------------------------------------------------
# Loss / Metrics
# ---------------------------------------------------------------------------

ce_loss = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()


def patch_attn_loss(logits, patch_features, labels, supcon_weight):
    return ce_loss(logits, labels) + supcon_weight * supcon_loss(patch_features, labels)


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


def run_eval(model, loader, desc):
    all_labels, all_probs = [], []
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits, _ = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 80

    train_df, val_df = prepare_splits(args.manifest, args.root_dir, val_ratio=args.val_ratio)
    train_dataset = ManifestImageDataset(train_df, args.root_dir)
    val_dataset = ManifestImageDataset(val_df, args.root_dir)
    cdf_dataset = CDFv1Dataset(args.cdf_csv, args.cdf_root)

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

    model = PatchAttnViT().to(device)
    check_layerscale(model.vit.base_model.model)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location="cpu"))
        print(f"Loaded checkpoint from {args.load_from}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler()

    lr_base = args.lr
    epochs = args.epochs
    iter_per_epoch = len(train_loader)
    total_steps = epochs * iter_per_epoch
    warmup_steps = args.warmup_steps
    lr_min = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmup_steps) * math.pi / (total_steps - warmup_steps))) / 2) + lr_min)
            if i > warmup_steps
            else (i / warmup_steps + lr_min)
        )
        for i in range(total_steps)
    }

    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[step]
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

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, patch_features = model(imgs)
                loss = patch_attn_loss(logits, patch_features, labels, args.supcon_weight)

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

            if batch_idx % 256 == 0:
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
        live_module.vit.save_pretrained(os.path.join(args.save_root, "latest_lora"))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            live_module.vit.save_pretrained(os.path.join(args.save_root, "best_lora"))
            print(f"\n  New best Test AUC={best_test_auc:.4f} -> saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Training complete. Best checkpoint: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
