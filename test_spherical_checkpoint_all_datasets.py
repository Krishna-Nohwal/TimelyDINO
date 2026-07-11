"""
Frame-level smoke/eval script for a spherical_cls checkpoint across all known
datasets. Missing datasets are reported and skipped.

Examples:
  python test_spherical_checkpoint_all_datasets.py \
      --checkpoint checkpoints_spherical_cls_gen_ft/best.pth

  python test_spherical_checkpoint_all_datasets.py \
      --checkpoint checkpoints_spherical_cls_gen_ft/best_resume.pth \
      --max_per_class 64
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
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

from augmentations import load_and_resize, normalize


IMG_SIZE = 256
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a spherical_cls checkpoint on small subsets of all available datasets."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--max_per_class", default=128, type=int,
                        help="Max real and max fake frames per dataset. 0 uses all discovered frames.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sphere_dim", default=0, type=int,
                        help="0 tries to infer from checkpoint, then falls back to 512.")
    parser.add_argument("--sphere_scale", default=16.0, type=float)
    parser.add_argument("--lora_r", default=0, type=int,
                        help="0 tries to infer from checkpoint, then falls back to 32.")
    parser.add_argument("--lora_alpha", default=64, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--ffpp_manifest", default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv")
    parser.add_argument("--ffpp_root", default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out")
    parser.add_argument("--cdfv1_csv", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv")
    parser.add_argument("--cdfv1_root", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out")
    parser.add_argument("--cdfv2_fake_root", default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2")
    parser.add_argument("--cdfv2_real_root", default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real")
    parser.add_argument("--cdfv3_root", default="/media/tarun/B482367C823642E2/usr/cdfv3_face_crops")
    parser.add_argument("--cdfv3_csv", default="/media/tarun/B482367C823642E2/usr/cdfv3_face_crops/manifest_cdfv3_face_crops.csv")
    parser.add_argument("--df0_fake_root", default="/media/tarun/B482367C823642E2/usr/df1.0_faces/fake")
    parser.add_argument("--df0_real_root", default="/media/tarun/B482367C823642E2/usr/df1.0_faces/real")
    parser.add_argument("--dfd_fake_root", default="/media/tarun/B482367C823642E2/usr/dfd_faces/fake")
    parser.add_argument("--dfd_real_root", default="/media/tarun/B482367C823642E2/usr/dfd_faces/real")
    parser.add_argument("--dfdc_fake_root", default="/media/tarun/B482367C823642E2/usr/dfdc/fake")
    parser.add_argument("--dfdc_real_root", default="/media/tarun/B482367C823642E2/usr/dfdc/real")
    parser.add_argument("--wdf_fake_root", default="/media/tarun/B482367C823642E2/usr/wdf/test/fake")
    parser.add_argument("--wdf_real_root", default="/media/tarun/B482367C823642E2/usr/wdf/test/real")
    parser.add_argument("--uadfv_fake_root", default="/media/tarun/B482367C823642E2/usr/uadfv_faces/fake")
    parser.add_argument("--uadfv_real_root", default="/media/tarun/B482367C823642E2/usr/uadfv_faces/real")
    parser.add_argument("--gen_root", default="/media/tarun/B482367C823642E2/usr/gen")
    parser.add_argument("--dvf_root", default="/media/tarun/B482367C823642E2/usr/dvf/DVF_recons_tiny")
    return parser.parse_args()


def simplex_prototypes(num_classes: int, dim: int) -> torch.Tensor:
    eye = torch.eye(num_classes, dtype=torch.float32)
    simplex = eye - eye.mean(dim=0, keepdim=True)
    simplex = nn.functional.normalize(simplex, dim=1)
    if dim > num_classes:
        simplex = torch.cat([simplex, torch.zeros(num_classes, dim - num_classes)], dim=1)
    return simplex


class FixedSphericalHead(nn.Module):
    def __init__(self, in_dim=1024, sphere_dim=512, num_classes=2, init_scale=16.0):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, sphere_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(sphere_dim, sphere_dim),
        )
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        self.register_buffer("prototypes", simplex_prototypes(num_classes, sphere_dim))

    def forward(self, x):
        z = self.projector(x.float())
        z = nn.functional.normalize(z, dim=1)
        prototypes = nn.functional.normalize(self.prototypes.float(), dim=1)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        return scale * (z @ prototypes.T), z


class SphericalCLSViT(nn.Module):
    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_CLASSES = 2

    def __init__(self, sphere_dim=512, sphere_scale=16.0, lora_r=32, lora_alpha=64, lora_dropout=0.0):
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
        return self.spherical_head(cls)


def clean_state_dict(state):
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_checkpoint_state(path: str):
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

    train_args = raw.get("args", {}) if isinstance(raw, dict) else {}
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw
    return ckpt_path, clean_state_dict(state), train_args


def infer_sphere_dim(state, requested):
    if requested > 0:
        return requested
    for key in ("spherical_head.prototypes", "spherical_head.projector.0.weight"):
        if key in state:
            return int(state[key].shape[-1] if key.endswith("prototypes") else state[key].shape[0])
    return 512


def infer_lora_r(state, requested):
    if requested > 0:
        return requested
    for key, value in state.items():
        if ".lora_A." in key and key.endswith(".weight"):
            return int(value.shape[0])
    return 32


def safe_walk_images(root: Path, image_png_only=False):
    paths = []
    if not root.is_dir():
        return paths
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        kept_dirs = []
        for dirname in dirnames:
            child = Path(dirpath) / dirname
            try:
                if not child.is_symlink():
                    kept_dirs.append(dirname)
            except OSError:
                pass
        dirnames[:] = kept_dirs
        for filename in filenames:
            path = Path(dirpath) / filename
            if image_png_only:
                if filename == "image.png" and path.is_file():
                    paths.append(str(path))
            elif path.suffix.lower() in IMAGE_EXTS and path.is_file():
                paths.append(str(path))
    return sorted(paths)


def add_item(items, path, label, dataset):
    items.append({"path": str(path), "label": int(label), "dataset": dataset})


def build_manifest_items(name, manifest_csv, root_dir, cdfv3_label_flip=False):
    if not manifest_csv or not root_dir:
        return [], f"{name}: manifest/root not provided"
    manifest = Path(manifest_csv)
    root = Path(root_dir)
    if not manifest.is_file():
        return [], f"{name}: missing manifest {manifest}"
    if not root.is_dir():
        return [], f"{name}: missing root {root}"

    df = pd.read_csv(manifest, sep=None, engine="python")
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        return [], f"{name}: manifest missing {required}, found {list(df.columns)}"

    items, skipped = [], 0
    for rel, raw_label in zip(df["sample_dir"].astype(str).str.replace("\\", "/", regex=False), df["label"]):
        label = int(raw_label)
        if cdfv3_label_flip:
            label = 0 if label == 1 else 1
        rel_path = Path(rel)
        path = rel_path / "image.png" if rel_path.is_absolute() else root / rel / "image.png"
        if path.is_file():
            add_item(items, path, label, name)
        else:
            skipped += 1
    note = f"{name}: loaded {len(items)} frames"
    if skipped:
        note += f", skipped {skipped} missing image.png"
    return items, note


def build_cdfv2_items(fake_root, real_root):
    items = []
    notes = []
    for root_str, label in [(real_root, 0), (fake_root, 1)]:
        root = Path(root_str) if root_str else Path("")
        if not root.is_dir():
            notes.append(f"missing {'real' if label == 0 else 'fake'} root {root_str}")
            continue
        for child in sorted(root.iterdir()):
            path = child / "image.png"
            if child.is_dir() and path.is_file():
                add_item(items, path, label, "CDFv2")
    return items, "CDFv2: " + (f"loaded {len(items)} frames" if items else "; ".join(notes))


def build_pair_items(name, fake_root, real_root, mode):
    items = []
    notes = []
    for root_str, label in [(real_root, 0), (fake_root, 1)]:
        root = Path(root_str) if root_str else Path("")
        if not root.is_dir():
            notes.append(f"missing {'real' if label == 0 else 'fake'} root {root_str}")
            continue
        if mode == "nested_image":
            paths = safe_walk_images(root, image_png_only=True)
        elif mode == "flat":
            paths = [
                str(path) for path in sorted(root.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
        else:
            paths = safe_walk_images(root, image_png_only=False)
        for path in paths:
            add_item(items, path, label, name)
    return items, f"{name}: " + (f"loaded {len(items)} frames" if items else "; ".join(notes))


def build_gen_items(gen_root):
    root = Path(gen_root) if gen_root else Path("")
    if not root.is_dir():
        return [], f"GEN: missing root {gen_root}"

    items, skipped = [], 0
    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        split_dirs = [d for d in [source_dir / "train", source_dir / "val"] if d.is_dir()]
        if not split_dirs:
            split_dirs = [source_dir]
        for split_dir in split_dirs:
            for class_name, label in [("nature", 0), ("ai", 1)]:
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    skipped += 1
                    continue
                dataset = f"GEN_{source_dir.name}_{split_dir.name}"
                for path in safe_walk_images(class_dir, image_png_only=False):
                    add_item(items, path, label, dataset)
    note = f"GEN: loaded {len(items)} frames"
    if skipped:
        note += f", skipped {skipped} missing nature/ai folders"
    return items, note


def build_dvf_items(dvf_root):
    root = Path(dvf_root) if dvf_root else Path("")
    if not root.is_dir():
        return [], f"DVF: missing root {dvf_root}"

    items, skipped = [], 0
    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        for class_name, label in [("0_real", 0), ("1_fake", 1)]:
            original_dir = source_dir / class_name / "original"
            if not original_dir.is_dir():
                skipped += 1
                continue
            dataset = f"DVF_{source_dir.name}"
            for path in safe_walk_images(original_dir, image_png_only=False):
                add_item(items, path, label, dataset)
    note = f"DVF: loaded {len(items)} frames"
    if skipped:
        note += f", skipped {skipped} missing original folders"
    return items, note


def collect_all_datasets(args):
    builders = [
        ("FFPP", lambda: build_manifest_items("FFPP", args.ffpp_manifest, args.ffpp_root)),
        ("CDFv1", lambda: build_manifest_items("CDFv1", args.cdfv1_csv, args.cdfv1_root)),
        ("CDFv2", lambda: build_cdfv2_items(args.cdfv2_fake_root, args.cdfv2_real_root)),
        ("CDF++", lambda: build_manifest_items("CDF++", args.cdfv3_csv, args.cdfv3_root, cdfv3_label_flip=True)),
        ("DF0", lambda: build_pair_items("DF0", args.df0_fake_root, args.df0_real_root, "nested_image")),
        ("DFD", lambda: build_pair_items("DFD", args.dfd_fake_root, args.dfd_real_root, "nested_image")),
        ("DFDC", lambda: build_pair_items("DFDC", args.dfdc_fake_root, args.dfdc_real_root, "flat")),
        ("WDF", lambda: build_pair_items("WDF", args.wdf_fake_root, args.wdf_real_root, "flat")),
        ("UADFV", lambda: build_pair_items("UADFV", args.uadfv_fake_root, args.uadfv_real_root, "nested_image")),
        ("GEN", lambda: build_gen_items(args.gen_root)),
        ("DVF", lambda: build_dvf_items(args.dvf_root)),
    ]

    dataset_items = {}
    print("\nDataset discovery:")
    for name, build_fn in builders:
        try:
            items, note = build_fn()
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        if items:
            dataset_items[name] = items
            counts = count_labels(items)
            print(f"  [use]  {note} | real={counts[0]} fake={counts[1]}")
        else:
            print(f"  [skip] {note}")
    return dataset_items


def count_labels(items):
    counts = defaultdict(int)
    for item in items:
        counts[int(item["label"])] += 1
    return counts


def sample_subset(items, max_per_class, seed):
    if max_per_class <= 0:
        return list(items)
    rng = np.random.default_rng(seed)
    sampled = []
    for label in [0, 1]:
        class_items = [item for item in items if int(item["label"]) == label]
        if len(class_items) > max_per_class:
            idx = rng.choice(len(class_items), size=max_per_class, replace=False)
            class_items = [class_items[int(i)] for i in idx]
        sampled.extend(class_items)
    order = rng.permutation(len(sampled)) if sampled else []
    return [sampled[int(i)] for i in order]


def stable_seed(base_seed, name):
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    return int(base_seed + offset)


class FrameDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = normalize(load_and_resize(item["path"], IMG_SIZE))
        return img, int(item["label"]), item["path"]


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int64)
    out = {
        "n": int(len(labels)),
        "real": int((labels == 0).sum()),
        "fake": int((labels == 1).sum()),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(set(labels.tolist())) == 2:
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        out["eer"] = float((fpr[idx] + fnr[idx]) / 2)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        out.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan
        out["eer"] = np.nan
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        out.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    return out


def evaluate_dataset(model, items, args, device, name):
    subset = sample_subset(items, args.max_per_class, stable_seed(args.seed, name))
    loader = DataLoader(
        FrameDataset(subset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    labels, probs = [], []
    amp_enabled = device.type == "cuda" and not args.no_amp
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
        for imgs, batch_labels, _ in tqdm(loader, desc=f"eval {name}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits, _ = model(imgs)
            batch_probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())
            labels.extend(batch_labels.numpy().tolist())
    return compute_metrics(labels, probs)


def format_metric(value, pct=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.2f}" if pct else f"{value:.4f}"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ckpt_path, state, train_args = load_checkpoint_state(args.checkpoint)
    sphere_dim = infer_sphere_dim(state, args.sphere_dim)
    lora_r = infer_lora_r(state, args.lora_r)
    lora_alpha = int(train_args.get("lora_alpha", args.lora_alpha)) if isinstance(train_args, dict) else args.lora_alpha

    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model config: sphere_dim={sphere_dim}, lora_r={lora_r}, lora_alpha={lora_alpha}")

    dataset_items = collect_all_datasets(args)
    if not dataset_items:
        raise SystemExit("No datasets available. Everything was skipped.")

    model = SphericalCLSViT(
        sphere_dim=sphere_dim,
        sphere_scale=args.sphere_scale,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\nLoaded weights: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")

    print("\nEvaluation subsets:")
    results = []
    for name, items in sorted(dataset_items.items()):
        subset = sample_subset(items, args.max_per_class, stable_seed(args.seed, name))
        counts = count_labels(subset)
        print(f"  {name:<8} using {len(subset):>5} frames | real={counts[0]:>5} fake={counts[1]:>5}")

    print("\nResults:")
    print("  dataset     n   real  fake     AUC      AP     Acc      F1     EER    TP    FP    FN    TN")
    for name, items in sorted(dataset_items.items()):
        metrics = evaluate_dataset(model, items, args, device, name)
        results.append((name, metrics))
        print(
            f"  {name:<8} {metrics['n']:>5} {metrics['real']:>5} {metrics['fake']:>5} "
            f"{format_metric(metrics['auc']):>7} {format_metric(metrics['ap']):>7} "
            f"{format_metric(metrics['acc'], pct=True):>7} {format_metric(metrics['f1']):>7} "
            f"{format_metric(metrics['eer'], pct=True):>7} "
            f"{metrics['tp']:>5} {metrics['fp']:>5} {metrics['fn']:>5} {metrics['tn']:>5}"
        )

    valid_aucs = [m["auc"] for _, m in results if not np.isnan(m["auc"])]
    if valid_aucs:
        print(f"\nMean dataset AUC over {len(valid_aucs)} two-class datasets: {float(np.mean(valid_aucs)):.4f}")


if __name__ == "__main__":
    main()
