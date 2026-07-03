"""
Evaluate the 4-layer frame-level model on real and fake image roots.

The real root is intended to match the usual FF++/ONCT preprocessed image
layout and only files literally named "image.png" are used. The fake root is
provided by the user and is scanned recursively for any supported image file
hidden anywhere inside the directory.

Example:
    python test_frame_model_real_fake_roots.py \
        --fake_root /path/to/fake/images \
        --real_root /raid/krishna/ffpp/preprocessed_out/real \
        --checkpoint main_checkpoint/best.pth
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from frame_model import ViT


IMG_SIZE = 256
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
LAYER_NAMES = ["layer20", "layer21", "layer22", "layer23"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test frame_model.py::ViT on real-root and fake-root images."
    )
    parser.add_argument("--real_root", default="/raid/krishna/ffpp/preprocessed_out/real", type=str)
    parser.add_argument("--fake_root", required=True, type=str)
    parser.add_argument("--checkpoint", default="main_checkpoint/best.pth", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--max_real", default=0, type=int, help="0 means use all discovered real images.")
    parser.add_argument("--max_fake", default=0, type=int, help="0 means use all discovered fake images.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_csv", default="", type=str, help="Optional per-image prediction CSV.")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def _safe_walk(root: str):
    """
    Yield files under root without crashing on dead directories, broken mount
    entries, permission errors, or folders that disappear during traversal.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Root does not exist: {root_path}")

    skipped_dirs = []

    def onerror(exc):
        skipped_dirs.append((getattr(exc, "filename", ""), repr(exc)))

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True, onerror=onerror, followlinks=False):
        # Avoid descending into symlinked dirs; they are common sources of loops
        # and dead entries on mounted/generated datasets.
        kept_dirs = []
        for dirname in dirnames:
            child = Path(dirpath) / dirname
            try:
                if child.is_symlink():
                    skipped_dirs.append((str(child), "symlink directory skipped"))
                    continue
            except OSError as exc:
                skipped_dirs.append((str(child), repr(exc)))
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                if path.is_file():
                    yield path, skipped_dirs
            except OSError as exc:
                skipped_dirs.append((str(path), repr(exc)))

    if skipped_dirs:
        yield None, skipped_dirs


def _print_skipped_scan_items(name: str, skipped: list[tuple[str, str]], max_print: int = 10):
    if not skipped:
        return
    print(f"  [{name}] skipped inaccessible/dead entries: {len(skipped)}")
    for path, reason in skipped[:max_print]:
        print(f"    skip: {path}  ({reason})")
    if len(skipped) > max_print:
        print(f"    ... {len(skipped) - max_print} more skipped")


def discover_real_images(root: str) -> list[str]:
    paths = []
    skipped = []
    for path, skipped_dirs in _safe_walk(root):
        skipped = skipped_dirs
        if path is not None and path.name == "image.png":
            paths.append(str(path))
    _print_skipped_scan_items("REAL scan", skipped)
    return sorted(paths)


def discover_fake_images(root: str) -> list[str]:
    paths = []
    skipped = []
    for path, skipped_dirs in _safe_walk(root):
        skipped = skipped_dirs
        if path is not None and path.suffix.lower() in IMAGE_EXTS:
            paths.append(str(path))
    _print_skipped_scan_items("FAKE scan", skipped)
    return sorted(paths)


def limit_paths(paths: list[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=limit, replace=False)
    return [paths[i] for i in sorted(indices)]


def summarize_paths(name: str, root: str, paths: list[str], sample_n: int = 8):
    print(f"\n{name} root: {root}")
    print(f"  discovered images: {len(paths)}")
    if not paths:
        return

    ext_counts = Counter(Path(p).suffix.lower() for p in paths)
    parent_counts = Counter(str(Path(p).parent.relative_to(root)) if str(Path(p)).startswith(str(Path(root))) else str(Path(p).parent) for p in paths)
    print(f"  extensions: {dict(ext_counts)}")
    print("  first images:")
    for p in paths[:sample_n]:
        print(f"    {p}")
    print("  common image parent dirs:")
    for parent, count in parent_counts.most_common(5):
        print(f"    {parent}: {count}")


def summarize_image_sizes(paths: list[str], name: str, sample_n: int = 256):
    if not paths:
        return
    sampled = paths[:sample_n]
    sizes = []
    bad = 0
    for p in sampled:
        try:
            with Image.open(p) as img:
                sizes.append(img.size)
        except Exception:
            bad += 1
    counts = Counter(sizes)
    print(f"  {name} size check over first {len(sampled)} images: bad={bad}")
    for size, count in counts.most_common(5):
        print(f"    {size[0]}x{size[1]}: {count}")


class RootImageDataset(Dataset):
    def __init__(self, real_paths: list[str], fake_paths: list[str]):
        self.entries = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        try:
            image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            ok = True
        except Exception:
            tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            ok = False
        return tensor, int(label), path, ok


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj))) if isinstance(obj, dict) else obj
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_model(checkpoint: str, device: torch.device):
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ViT()
    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    return model


def compute_metrics(labels, probs, name: str):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    print(
        f"\n[{name}] AUC={auc:.4f}  AP={ap:.4f}  Acc={acc*100:.2f}%  "
        f"F1={f1:.4f}  EER={eer*100:.2f}%"
    )
    print(
        f"  TPR={tpr*100:.2f}%  FPR={fpr*100:.2f}%  TNR={tnr*100:.2f}%  "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    print(
        f"  P(fake) real: mean={real_probs.mean():.4f} std={real_probs.std():.4f} "
        f"q05={np.quantile(real_probs, 0.05):.4f} median={np.median(real_probs):.4f} q95={np.quantile(real_probs, 0.95):.4f}"
    )
    print(
        f"  P(fake) fake: mean={fake_probs.mean():.4f} std={fake_probs.std():.4f} "
        f"q05={np.quantile(fake_probs, 0.05):.4f} median={np.median(fake_probs):.4f} q95={np.quantile(fake_probs, 0.95):.4f}"
    )


def run_eval(model, loader, device: torch.device, use_amp: bool):
    all_labels = []
    all_paths = []
    all_ok = []
    layer_probs = [[] for _ in LAYER_NAMES]

    amp_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)
    with torch.inference_mode(), amp_context:
        for imgs, labels, paths, ok in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits_list, _, _ = model(imgs)
            for idx, logits in enumerate(logits_list):
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
                layer_probs[idx].extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(list(paths))
            all_ok.extend(ok.numpy().astype(bool).tolist())

    return all_labels, layer_probs, all_paths, all_ok


def save_predictions(path: str, image_paths: list[str], labels: list[int], layer_probs: list[list[float]], ok: list[bool]):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["path", "label", "ok"] + [f"{name}_p_fake" for name in LAYER_NAMES] + ["mean_p_fake"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, image_path in enumerate(image_paths):
            row = {"path": image_path, "label": labels[i], "ok": ok[i]}
            vals = [layer_probs[j][i] for j in range(len(LAYER_NAMES))]
            for name, val in zip(LAYER_NAMES, vals):
                row[f"{name}_p_fake"] = f"{val:.8f}"
            row["mean_p_fake"] = f"{float(np.mean(vals)):.8f}"
            writer.writerow(row)
    print(f"\nSaved per-image predictions: {csv_path}")


def print_error_examples(labels, probs, paths, title: str, top_k: int = 10):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    high_real = real_idx[np.argsort(-probs[real_idx])[:top_k]]
    low_fake = fake_idx[np.argsort(probs[fake_idx])[:top_k]]

    print(f"\nMost fake-looking REAL images by {title}:")
    for idx in high_real:
        print(f"  p_fake={probs[idx]:.4f}  {paths[idx]}")

    print(f"\nMost real-looking FAKE images by {title}:")
    for idx in low_fake:
        print(f"  p_fake={probs[idx]:.4f}  {paths[idx]}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"Using device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

    real_paths = discover_real_images(args.real_root)
    fake_paths = discover_fake_images(args.fake_root)
    real_paths = limit_paths(real_paths, args.max_real, args.seed)
    fake_paths = limit_paths(fake_paths, args.max_fake, args.seed + 1)

    summarize_paths("REAL", args.real_root, real_paths)
    summarize_image_sizes(real_paths, "REAL")
    summarize_paths("FAKE", args.fake_root, fake_paths)
    summarize_image_sizes(fake_paths, "FAKE")

    if not real_paths or not fake_paths:
        raise ValueError("Need at least one real image and one fake image.")

    print("\nEvaluation set:")
    print(f"  real images: {len(real_paths)}")
    print(f"  fake images: {len(fake_paths)}")
    print(f"  total images: {len(real_paths) + len(fake_paths)}")
    print(f"  real/fake ratio: {len(real_paths) / max(len(fake_paths), 1):.4f}")

    dataset = RootImageDataset(real_paths, fake_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(args.checkpoint, device)
    labels, layer_probs, paths, ok = run_eval(model, loader, device, use_amp)
    bad_count = len(ok) - int(np.sum(ok))
    if bad_count:
        print(f"\nWARNING: {bad_count} images failed to load and were evaluated as zeros.")
    else:
        print("\nAll images loaded successfully.")

    for name, probs in zip(LAYER_NAMES, layer_probs):
        compute_metrics(labels, probs, name)

    mean_probs = np.mean(np.asarray(layer_probs), axis=0)
    compute_metrics(labels, mean_probs, "mean_of_4_layers")
    print_error_examples(labels, layer_probs[-1], paths, "layer23")

    if args.save_csv:
        save_predictions(args.save_csv, paths, labels, layer_probs, ok)


if __name__ == "__main__":
    main()
