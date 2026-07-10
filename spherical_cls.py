"""
Spherical CLS-only Stage 1 training script.

This is a spherical variant of cls.py. It uses both real and fake labels,
trains on a combined multi-dataset frame pool, and still uses only the
final-layer DINOv3 CLS token. The learnable linear classifier is replaced with
fixed simplex prototypes on the unit hypersphere:

  cls_last -> projector -> L2 normalize -> cosine logits to fixed prototypes
  loss     = CrossEntropy(logits, labels) + lambda * SupCon(sphere_features, labels)

For the binary real/fake case, the simplex prototypes are antipodal anchors on
the sphere. CDFv1 remains the held-out per-epoch test set.
"""

import argparse
import math
import os
from collections import defaultdict
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
    description="Stage 1 spherical CLS training: final-layer CLS + fixed simplex prototypes"
)
parser.add_argument("--epochs",        default=50,   type=int)
parser.add_argument("--batch_size",    default=128,  type=int)
parser.add_argument("--num_workers",   default=36,   type=int)
parser.add_argument("--save_root",     default="checkpoints_spherical_cls_ft", type=str)
parser.add_argument("--load_from",     default="checkpoints_spherical_cls/best.pth",   type=str)
parser.add_argument("--no_resume_state", action="store_true",
                    help="Load model weights only from --load_from, ignoring optimizer/scheduler/scaler state.")
parser.add_argument("--manifest", type=str)
parser.add_argument("--root_dir", type=str)
parser.add_argument("--cdf_root",      default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv",       default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
parser.add_argument("--val_ratio",     default=0.005, type=float,
                    help="Tiny per-dataset frame-level validation split.")
parser.add_argument("--lr",            default=1e-4, type=float)
parser.add_argument("--warmup_steps",  default=512,  type=int)
parser.add_argument("--supcon_weight", default=1/16, type=float)
parser.add_argument("--sphere_dim",    default=512,  type=int)
parser.add_argument("--sphere_scale",  default=16.0, type=float)
parser.add_argument("--max_train_samples", default=0, type=int,
                    help="Optional cap on training images. 0 uses all training images.")
parser.add_argument("--max_val_samples", default=0, type=int,
                    help="Optional cap on validation images. 0 uses all validation images.")
parser.add_argument("--max_frames_per_dataset", default=0, type=int,
                    help="Optional stratified train-frame cap per dataset. 0 uses all frames.")
parser.add_argument("--seed", default=42, type=int)

# Extra training datasets from the UMAP scripts.
parser.add_argument("--cdfv2_fake_root", type=str)
parser.add_argument("--cdfv2_real_root", type=str)
parser.add_argument("--cdfv3_root",      default="/raid/krishna/cdfv3_face_crops", type=str)
parser.add_argument("--cdfv3_csv",       default="/raid/krishna/cdfv3_face_crops/manifest_cdfv3_face_crops.csv", type=str)
parser.add_argument("--df0_fake_root",   default="/raid/krishna/df1.0_faces/fake", type=str)
parser.add_argument("--df0_real_root",   default="/raid/krishna/df1.0_faces/real", type=str)
parser.add_argument("--dfo_fake_root",   default="", type=str)
parser.add_argument("--dfo_real_root",   default="", type=str)
parser.add_argument("--dfd_fake_root",   default="/raid/krishna/dfd_faces/fake", type=str)
parser.add_argument("--dfd_real_root",   default="/raid/krishna/dfd_faces/real", type=str)
parser.add_argument("--dfdc_fake_root",  default="/raid/krishna/dfdc/fake", type=str)
parser.add_argument("--dfdc_real_root",  default="/raid/krishna/dfdc/real", type=str)
parser.add_argument("--wdf_fake_root",   default="/raid/krishna/wdf/test/fake", type=str)
parser.add_argument("--wdf_real_root",   default="/raid/krishna/wdf/test/real", type=str)
parser.add_argument("--uadfv_fake_root", default="/raid/krishna/uadfv_faces/fake", type=str)
parser.add_argument("--uadfv_real_root", default="/raid/krishna/uadfv_faces/real", type=str)
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

def simplex_prototypes(num_classes: int, dim: int) -> torch.Tensor:
    """
    Returns fixed unit vectors with equal pairwise cosine -1 / (C - 1).

    For C=2, this gives antipodal real/fake anchors. The minimal regular
    simplex lives in C dimensions here with rank C-1, then is zero-padded to
    the requested embedding dimension.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if dim < num_classes:
        raise ValueError("sphere_dim must be >= num_classes")

    eye = torch.eye(num_classes, dtype=torch.float32)
    simplex = eye - eye.mean(dim=0, keepdim=True)
    simplex = nn.functional.normalize(simplex, dim=1)

    if dim > num_classes:
        pad = torch.zeros(num_classes, dim - num_classes, dtype=torch.float32)
        simplex = torch.cat([simplex, pad], dim=1)

    return simplex


class FixedSphericalHead(nn.Module):
    """Project CLS features onto a hypersphere and score fixed simplex anchors."""

    def __init__(
        self,
        in_dim: int = 1024,
        sphere_dim: int = 512,
        num_classes: int = 2,
        init_scale: float = 16.0,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, sphere_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(sphere_dim, sphere_dim),
        )
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        self.register_buffer("prototypes", simplex_prototypes(num_classes, sphere_dim))

    def forward(self, x: torch.Tensor):
        z = self.projector(x.float())
        z = nn.functional.normalize(z, dim=1)
        prototypes = nn.functional.normalize(self.prototypes.float(), dim=1)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        logits = scale * (z @ prototypes.T)
        return logits, z


class SphericalCLSViT(nn.Module):
    """DINOv3 ViT-L/16 using final-layer CLS and fixed spherical prototypes."""

    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_CLASSES = 2

    def __init__(self, sphere_dim: int = 512, sphere_scale: float = 16.0):
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
        self.spherical_head = FixedSphericalHead(
            self.EMBED_DIM,
            sphere_dim=sphere_dim,
            num_classes=self.NUM_CLASSES,
            init_scale=sphere_scale,
        )

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        _, prefix_tokens = intermediates[0]
        cls = prefix_tokens[:, 0, :]
        logits, sphere_features = self.spherical_head(cls)
        return logits, sphere_features


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

def cap_items_by_label(items: list[dict], cap: int, seed: int, split_name: str) -> list[dict]:
    if cap <= 0 or len(items) <= cap:
        return items

    rng = np.random.default_rng(seed)
    reals = [item for item in items if item["label"] == 0]
    fakes = [item for item in items if item["label"] == 1]
    if not reals or not fakes:
        idx = rng.choice(len(items), size=cap, replace=False)
        capped = [items[int(i)] for i in idx]
    else:
        n_real = min(len(reals), cap // 2)
        n_fake = min(len(fakes), cap - n_real)
        n_real = min(len(reals), cap - n_fake)
        real_idx = rng.choice(len(reals), size=n_real, replace=False)
        fake_idx = rng.choice(len(fakes), size=n_fake, replace=False)
        capped = [reals[int(i)] for i in real_idx] + [fakes[int(i)] for i in fake_idx]

    order = rng.permutation(len(capped))
    capped = [capped[int(i)] for i in order]
    print_dataset_item_counts(f"{split_name} capped", capped)
    return capped


def split_items_by_label(items: list[dict], val_ratio: float, seed: int):
    if val_ratio <= 0:
        return items, []

    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(set(item["label"] for item in items)):
        class_items = [item for item in items if item["label"] == label]
        if len(class_items) <= 1:
            train.extend(class_items)
            continue
        order = rng.permutation(len(class_items))
        class_items = [class_items[int(i)] for i in order]
        val_n = max(1, int(len(class_items) * val_ratio))
        val_n = min(val_n, len(class_items) - 1)
        val.extend(class_items[:val_n])
        train.extend(class_items[val_n:])
    return train, val


def print_dataset_item_counts(title: str, items: list[dict]):
    print(f"\n{title}:")
    by_dataset = defaultdict(lambda: [0, 0, 0])
    for item in items:
        counts = by_dataset[item["dataset"]]
        counts[0] += 1
        counts[1 + int(item["label"])] += 1
    total = [0, 0, 0]
    for dataset in sorted(by_dataset):
        counts = by_dataset[dataset]
        total = [a + b for a, b in zip(total, counts)]
        print(f"  {dataset:<8} frames={counts[0]:>8} real={counts[1]:>8} fake={counts[2]:>8}")
    print(f"  {'TOTAL':<8} frames={total[0]:>8} real={total[1]:>8} fake={total[2]:>8}")


def build_ffpp_items(manifest_csv: str, root_dir: str):
    if not manifest_csv or not root_dir:
        print("  [skip] FFPP: --manifest / --root_dir not provided.")
        return []
    if not Path(manifest_csv).is_file():
        print(f"  [skip] FFPP: missing manifest {manifest_csv}")
        return []
    if not Path(root_dir).is_dir():
        print(f"  [skip] FFPP: missing root {root_dir}")
        return []

    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    root = Path(root_dir)

    items, skipped = [], 0
    for rel, label in zip(df["sample_dir"].astype(str).str.replace("\\", "/", regex=False), df["label"]):
        rel_path = Path(rel)
        path = rel_path / "image.png" if rel_path.is_absolute() else root / rel / "image.png"
        if path.is_file():
            items.append({"path": str(path), "label": int(label), "dataset": "FFPP"})
        else:
            skipped += 1
    if skipped:
        print(f"  [FFPP] skipped {skipped} missing image.png files")
    print_dataset_item_counts("FFPP frames loaded", items)
    return items


def build_cdfv2_items(fake_root: str, real_root: str):
    if not fake_root or not real_root:
        print("  [skip] CDFv2: --cdfv2_fake_root / --cdfv2_real_root not provided.")
        return []
    items = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [CDFv2] WARNING: missing root {root}")
            continue
        for d in sorted(root.iterdir()):
            path = d / "image.png"
            if d.is_dir() and path.exists():
                items.append({"path": str(path), "label": label, "dataset": "CDFv2"})
    print_dataset_item_counts("CDFv2 frames loaded", items)
    return items


def build_cdfv3_items(cdfv3_csv: str, cdfv3_root: str):
    if not cdfv3_root:
        print("  [skip] CDFv3: --cdfv3_root not provided.")
        return []
    cdfv3_csv = cdfv3_csv or str(Path(cdfv3_root) / "manifest_cdfv3_face_crops.csv")
    if not Path(cdfv3_root).is_dir():
        print(f"  [skip] CDFv3: missing root {cdfv3_root}")
        return []
    if not Path(cdfv3_csv).is_file():
        print(f"  [skip] CDFv3: missing manifest {cdfv3_csv}")
        return []

    df = pd.read_csv(cdfv3_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CDFv3 manifest must contain {required}. Found: {list(df.columns)}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    root = Path(cdfv3_root)

    items, skipped = [], 0
    for rel, manifest_label in zip(df["sample_dir"].astype(str).str.replace("\\", "/", regex=False), df["label"]):
        label = 0 if int(manifest_label) == 1 else 1
        rel_path = Path(rel)
        path = rel_path / "image.png" if rel_path.is_absolute() else root / rel / "image.png"
        if path.is_file():
            items.append({"path": str(path), "label": label, "dataset": "CDFv3"})
        else:
            skipped += 1
    if skipped:
        print(f"  [CDFv3] skipped {skipped} missing image.png files")
    print_dataset_item_counts("CDFv3 frames loaded", items)
    return items


def build_nested_image_items(fake_root: str, real_root: str, dataset_name: str):
    if not fake_root or not real_root:
        print(f"  [skip] {dataset_name}: fake/real roots not provided.")
        return []
    items = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        for path in sorted(root.rglob("image.png")):
            if path.is_file():
                items.append({"path": str(path), "label": label, "dataset": dataset_name})
    print_dataset_item_counts(f"{dataset_name} frames loaded", items)
    return items


def build_flat_items(fake_root: str, real_root: str, dataset_name: str):
    if not fake_root or not real_root:
        print(f"  [skip] {dataset_name}: fake/real roots not provided.")
        return []
    items = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        skipped = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                skipped += 1
                continue
            items.append({"path": str(path), "label": label, "dataset": dataset_name})
        if skipped:
            print(f"  [{dataset_name}] skipped {skipped} files under {root}")
    print_dataset_item_counts(f"{dataset_name} frames loaded", items)
    return items


def build_dataset_items(args):
    print("\nBuilding training dataset frame lists ...")
    dataset_items = {
        "FFPP": build_ffpp_items(args.manifest, args.root_dir),
        "CDFv2": build_cdfv2_items(args.cdfv2_fake_root, args.cdfv2_real_root),
        "CDFv3": build_cdfv3_items(args.cdfv3_csv, args.cdfv3_root),
    }

    dfo_fake = args.dfo_fake_root or args.df0_fake_root
    dfo_real = args.dfo_real_root or args.df0_real_root
    dataset_items["DFo"] = build_nested_image_items(dfo_fake, dfo_real, "DFo")
    dataset_items["DFD"] = build_nested_image_items(args.dfd_fake_root, args.dfd_real_root, "DFD")
    dataset_items["DFDC"] = build_flat_items(args.dfdc_fake_root, args.dfdc_real_root, "DFDC")
    dataset_items["WDF"] = build_flat_items(args.wdf_fake_root, args.wdf_real_root, "WDF")
    dataset_items["UADFV"] = build_nested_image_items(args.uadfv_fake_root, args.uadfv_real_root, "UADFV")
    return {name: items for name, items in dataset_items.items() if items}


def prepare_splits(args):
    dataset_items = build_dataset_items(args)
    if not dataset_items:
        raise ValueError("No training frames found. Check dataset paths.")

    train_items, val_items = [], []
    print("\nFrame-level train/val split:")
    for offset, (dataset_name, items) in enumerate(sorted(dataset_items.items()), start=100):
        dataset_train, dataset_val = split_items_by_label(items, args.val_ratio, args.seed + offset + 1000)
        dataset_train = cap_items_by_label(
            dataset_train,
            args.max_frames_per_dataset,
            args.seed + offset + 2000,
            f"{dataset_name} train",
        )
        print(
            f"  {dataset_name:<8} total_frames={len(items):>8} "
            f"train_frames={len(dataset_train):>8} val_frames={len(dataset_val):>6}"
        )
        train_items.extend(dataset_train)
        val_items.extend(dataset_val)

    train_items = cap_items_by_label(train_items, args.max_train_samples, args.seed + 9999, "Combined train")
    val_items = cap_items_by_label(val_items, args.max_val_samples, args.seed + 10099, "Combined val")

    if not train_items:
        raise ValueError("No training frames found after split/caps.")
    if not val_items:
        raise ValueError("No validation frames found. Increase --val_ratio or check dataset paths.")

    rng = np.random.default_rng(args.seed)
    train_items = [train_items[int(i)] for i in rng.permutation(len(train_items))]
    val_items = [val_items[int(i)] for i in rng.permutation(len(val_items))]
    print_dataset_item_counts("Combined train frames", train_items)
    print_dataset_item_counts("Combined val frames", val_items)
    return train_items, val_items


class FrameItemDataset(Dataset):
    """Multi-dataset frame loader. label: 0=Real, 1=Fake."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = load_and_resize(item["path"], IMG_SIZE)
        img = normalize(img)
        return img, int(item["label"])


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


def spherical_cls_loss(logits, sphere_features, labels, supcon_weight):
    return ce_loss(logits, labels) + supcon_weight * supcon_loss(sphere_features, labels)


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


def clean_state_dict(state):
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_training_checkpoint(path: str):
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        resume_path = ckpt_path / "latest_resume.pth"
        best_path = ckpt_path / "best.pth"
        ckpt_path = resume_path if resume_path.is_file() else best_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        model_state = clean_state_dict(raw["model"])
        train_state = raw
    elif isinstance(raw, dict) and "state_dict" in raw:
        model_state = clean_state_dict(raw["state_dict"])
        train_state = raw
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        model_state = clean_state_dict(raw["model_state_dict"])
        train_state = raw
    else:
        model_state = clean_state_dict(raw)
        train_state = {}
    return ckpt_path, model_state, train_state


def make_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_test_auc: float,
    best_epoch: int,
):
    live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "epoch": epoch,
        "model": live_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_test_auc": best_test_auc,
        "best_epoch": best_epoch,
        "args": vars(args),
    }


def cdfv1_available(cdf_csv: str, cdf_root: str) -> bool:
    if not cdf_csv or not cdf_root:
        print("CDFv1 eval disabled: --cdf_csv / --cdf_root not provided.")
        return False
    if not Path(cdf_csv).is_file():
        print(f"CDFv1 eval disabled: missing manifest {cdf_csv}")
        return False
    if not Path(cdf_root).is_dir():
        print(f"CDFv1 eval disabled: missing root {cdf_root}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 80

    train_items, val_items = prepare_splits(args)
    train_dataset = FrameItemDataset(train_items)
    val_dataset = FrameItemDataset(val_items)
    cdf_dataset = CDFv1Dataset(args.cdf_csv, args.cdf_root) if cdfv1_available(args.cdf_csv, args.cdf_root) else None

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
    cdf_loader = None
    if cdf_dataset is not None:
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

    model = SphericalCLSViT(
        sphere_dim=args.sphere_dim,
        sphere_scale=args.sphere_scale,
    ).to(device)
    check_layerscale(model.vit.base_model.model)

    train_state = {}
    start_epoch = 0
    if args.load_from:
        ckpt_path, model_state, train_state = load_training_checkpoint(args.load_from)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Loaded model weights from {ckpt_path}")
        print(f"  missing keys   : {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler()

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

    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[min(step, total_steps - 1)]
    )

    best_test_auc = 0.0
    best_epoch = -1
    if train_state and not args.no_resume_state:
        if "optimizer" in train_state:
            try:
                optimizer.load_state_dict(train_state["optimizer"])
                print("Restored optimizer state.")
            except Exception as exc:
                print(f"Could not restore optimizer state: {exc}")
        if "scheduler" in train_state:
            try:
                scheduler.load_state_dict(train_state["scheduler"])
                print("Restored scheduler state.")
            except Exception as exc:
                print(f"Could not restore scheduler state: {exc}")
        if "scaler" in train_state:
            try:
                scaler.load_state_dict(train_state["scaler"])
                print("Restored AMP scaler state.")
            except Exception as exc:
                print(f"Could not restore AMP scaler state: {exc}")
        start_epoch = int(train_state.get("epoch", -1)) + 1
        best_test_auc = float(train_state.get("best_test_auc", 0.0))
        best_epoch = int(train_state.get("best_epoch", -1))
        print(f"Continuing training from epoch {start_epoch + 1}/{epochs}.")
    elif train_state and args.no_resume_state:
        print("Loaded model weights only; optimizer/scheduler/scaler state ignored.")

    if start_epoch >= epochs:
        print(f"Checkpoint epoch {start_epoch} is already >= --epochs {epochs}; no training epochs to run.")

    for epoch in range(start_epoch, epochs):
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
                logits, sphere_features = model(imgs)
                loss = spherical_cls_loss(logits, sphere_features, labels, args.supcon_weight)

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
        val_auc = compute_metrics(val_labels, val_probs, "Val  ", epoch)

        if cdf_loader is not None:
            cdf_labels, cdf_probs = run_eval(model, cdf_loader, f"Epoch {epoch+1} [CDFv1]")
            test_auc = compute_metrics(cdf_labels, cdf_probs, "Test ", epoch)
            score_name = "Test AUC"
        else:
            test_auc = val_auc
            score_name = "Val AUC"

        live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = live_module.state_dict()
        resume_state = make_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_test_auc,
            best_epoch,
        )

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            resume_state["best_test_auc"] = best_test_auc
            resume_state["best_epoch"] = best_epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            torch.save(resume_state, os.path.join(args.save_root, "best_resume.pth"))
            live_module.vit.save_pretrained(os.path.join(args.save_root, "best_lora"))
            print(f"\n  New best {score_name}={best_test_auc:.4f} -> saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  {score_name}={best_test_auc:.4f}")

        resume_state["best_test_auc"] = best_test_auc
        resume_state["best_epoch"] = best_epoch
        torch.save(state_dict, os.path.join(args.save_root, "latest.pth"))
        torch.save(resume_state, os.path.join(args.save_root, "latest_resume.pth"))
        live_module.vit.save_pretrained(os.path.join(args.save_root, "latest_lora"))

    print(f"\n{SEP}")
    print(f"  Training complete. Best checkpoint: epoch {best_epoch+1}  AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
