"""
Hyperspherical CLS-token training using the spherical component from CTrue.

This is not the projection-cosine + cross-entropy setup from spherical_cls.py.
The model still uses the final DINOv3 CLS token, but training uses the paper's
hyperspherical objective L_S:

  z_i^S = normalize(projector(CLS_i))
  L_S pulls z_i^S toward its regular-simplex class prototype and same-class
  batch samples, while contrasting against all other batch samples.

The paper obtains labels from self-supervised pseudo-fake classes. This script
uses the available real/fake labels as the supervised classes for the same
objective, which keeps it usable with the current frame datasets.
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
import transformers

if not hasattr(transformers, "HybridCache"):
    transformers.HybridCache = getattr(transformers, "DynamicCache", getattr(transformers, "Cache", object))

from peft import LoraConfig, get_peft_model
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


parser = argparse.ArgumentParser(
    description="DINOv3 final-CLS training with the CTrue hyperspherical loss L_S."
)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=36, type=int)
parser.add_argument("--save_root", default="checkpoints_hyperspherical_cls", type=str)
parser.add_argument("--load_from", default="checkpoints_spherical_cls/best.pth", type=str)
parser.add_argument("--no_compile", action="store_true")

parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup_steps", default=512, type=int)
parser.add_argument("--temperature", default=1.0, type=float,
                    help="Temperature for L_S. The paper omits tau, so 1.0 matches the written equation.")
parser.add_argument("--sphere_dim", default=512, type=int)
parser.add_argument("--lora_r", default=32, type=int)
parser.add_argument("--lora_alpha", default=64, type=int)
parser.add_argument("--lora_dropout", default=0.10, type=float)
parser.add_argument("--max_train_samples", default=0, type=int)
parser.add_argument("--max_val_samples", default=0, type=int)
parser.add_argument("--max_frames_per_dataset", default=0, type=int)
parser.add_argument("--val_ratio", default=0.005, type=float)
parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--manifest",
                    default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv",
                    type=str)
parser.add_argument("--root_dir",
                    default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out",
                    type=str)
parser.add_argument("--cdf_root", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
args = parser.parse_args()


IMG_SIZE = 256
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers
torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


def simplex_prototypes(num_classes: int, dim: int) -> torch.Tensor:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if dim < num_classes:
        raise ValueError("sphere_dim must be >= num_classes")
    eye = torch.eye(num_classes, dtype=torch.float32)
    simplex = eye - eye.mean(dim=0, keepdim=True)
    simplex = nn.functional.normalize(simplex, dim=1)
    if dim > num_classes:
        simplex = torch.cat([simplex, torch.zeros(num_classes, dim - num_classes)], dim=1)
    return simplex


class HypersphericalHead(nn.Module):
    """Project CLS features to S^d and hold fixed regular-simplex prototypes."""

    def __init__(self, in_dim=1024, sphere_dim=512, num_classes=2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, sphere_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(sphere_dim, sphere_dim),
        )
        self.register_buffer("prototypes", simplex_prototypes(num_classes, sphere_dim))

    def forward(self, x):
        z = self.projector(x.float())
        z = nn.functional.normalize(z, dim=1)
        prototypes = nn.functional.normalize(self.prototypes.float(), dim=1)
        proto_sims = z @ prototypes.T
        return proto_sims, z


class HypersphericalCLSViT(nn.Module):
    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_CLASSES = 2

    def __init__(self, sphere_dim=512, lora_r=32, lora_alpha=64, lora_dropout=0.10):
        super().__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["attn.qkv"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)
        # Keep the name spherical_head so projector weights can be reused from
        # old spherical_cls checkpoints, while the loss below is different.
        self.spherical_head = HypersphericalHead(
            in_dim=self.EMBED_DIM,
            sphere_dim=sphere_dim,
            num_classes=self.NUM_CLASSES,
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
        return self.spherical_head(cls)


def hyperspherical_ctru_loss(proto_sims, sphere_features, labels, temperature=1.0):
    """
    Paper Eq. (2), adapted to a minibatch.

    P(i): same-label samples other than i.
    N(i): all other samples in the batch.
    The prototype term uses p_yi; the sample term uses positives in P(i).
    If a class appears only once in the batch, we keep the prototype term.
    """
    if sphere_features.size(0) <= 1:
        raise ValueError("hyperspherical_ctru_loss needs batch_size > 1")

    z = nn.functional.normalize(sphere_features.float(), dim=1)
    labels = labels.long()
    bsz = z.size(0)
    eye = torch.eye(bsz, dtype=torch.bool, device=z.device)

    sample_logits = (z @ z.T) / max(float(temperature), 1e-8)
    sample_logits = sample_logits.masked_fill(eye, -torch.inf)
    log_den = torch.logsumexp(sample_logits, dim=1)

    proto_logprob = proto_sims.float().div(max(float(temperature), 1e-8)).gather(1, labels[:, None]).squeeze(1) - log_den

    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    positive_count = positive_mask.sum(dim=1)
    positive_logprob = (sample_logits - log_den[:, None]).masked_fill(~positive_mask, 0.0).sum(dim=1)

    has_positive = positive_count > 0
    exact_paper_term = -(proto_logprob + positive_logprob) / positive_count.clamp_min(1).float()
    proto_only_term = -proto_logprob
    return torch.where(has_positive, exact_paper_term, proto_only_term).mean()


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
        print(f"  {dataset:<12} frames={counts[0]:>8} real={counts[1]:>8} fake={counts[2]:>8}")
    print(f"  {'TOTAL':<12} frames={total[0]:>8} real={total[1]:>8} fake={total[2]:>8}")


def cap_items_by_label(items: list[dict], cap: int, seed: int, split_name: str) -> list[dict]:
    if cap <= 0 or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    capped = []
    for label in [0, 1]:
        class_items = [item for item in items if item["label"] == label]
        n = min(len(class_items), cap // 2 if label == 0 else cap - len(capped))
        if n > 0:
            idx = rng.choice(len(class_items), size=n, replace=False)
            capped.extend(class_items[int(i)] for i in idx)
    if not capped:
        idx = rng.choice(len(items), size=cap, replace=False)
        capped = [items[int(i)] for i in idx]
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


def build_ffpp_items(manifest_csv: str, root_dir: str):
    if not manifest_csv or not root_dir:
        raise ValueError("--manifest and --root_dir are required for FF++ training.")
    manifest = Path(manifest_csv)
    root = Path(root_dir)
    if not manifest.is_file():
        raise FileNotFoundError(f"FF++ manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"FF++ root not found: {root}")

    df = pd.read_csv(manifest)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")

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
    print_dataset_item_counts("FF++ frames loaded", items)
    return items


def build_dataset_items(args):
    dataset_items = {
        "FFPP": build_ffpp_items(args.manifest, args.root_dir),
    }
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
            f"  {dataset_name:<12} total_frames={len(items):>8} "
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
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return normalize(load_and_resize(item["path"], IMG_SIZE)), int(item["label"])


class CDFv1Dataset(Dataset):
    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = df["label"].astype(int)
        root = Path(data_root)
        paths = df["sample_dir"].apply(lambda d: str(root / d / "image.png"))
        labels = df["label"].values
        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [CDFv1] skipped {skipped} missing image.png")
        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))
        print(f"CDFv1 eval frames: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        return normalize(load_and_resize(img_path, IMG_SIZE)), int(label)


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
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f"  [{split_name}] Epoch {epoch+1:02d} | "
          f"AUC={auc:.4f} AP={ap:.4f} Acc={acc*100:.2f}% F1={f1:.4f} EER={eer*100:.2f}% "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return auc


def run_eval(model, loader, desc):
    labels_out, probs_out = [], []
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            proto_sims, _ = model(imgs)
            probs = torch.softmax(proto_sims.float(), dim=1)[:, 1].cpu().numpy()
            probs_out.extend(probs.tolist())
            labels_out.extend(labels.numpy().tolist())
    return labels_out, probs_out


def clean_state_dict(state):
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_training_checkpoint(path: str):
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        candidates = [
            ckpt_path / "best.pth",
            ckpt_path / "latest.pth",
            ckpt_path / "best_resume.pth",
            ckpt_path / "latest_resume.pth",
        ]
        ckpt_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
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


def cdfv1_available(cdf_csv: str, cdf_root: str) -> bool:
    return bool(cdf_csv and cdf_root and Path(cdf_csv).is_file() and Path(cdf_root).is_dir())


def make_resume_checkpoint(model, optimizer, scheduler, scaler, epoch, best_auc, best_epoch):
    live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "epoch": epoch,
        "model": live_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_test_auc": best_auc,
        "best_epoch": best_epoch,
        "args": vars(args),
        "loss": "CTrue hyperspherical L_S",
    }


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
        drop_last=True,
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
    model = HypersphericalCLSViT(
        sphere_dim=args.sphere_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

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
    total_steps = max(epochs * iter_per_epoch, 1)
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
    if train_state:
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

    for epoch in range(start_epoch, epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)
        model.train()
        train_labels, train_probs = [], []
        iter_i = epoch * iter_per_epoch

        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs = augment_batch(imgs)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                proto_sims, sphere_features = model(imgs)
                loss = hyperspherical_ctru_loss(
                    proto_sims,
                    sphere_features,
                    labels,
                    temperature=args.temperature,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                probs = torch.softmax(proto_sims.float(), dim=1)[:, 1].cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 256 == 0:
                print(f"  batch={batch_idx:4d}/{iter_per_epoch}  L_S={loss.item():.4f}")

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
