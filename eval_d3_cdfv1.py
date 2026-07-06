"""
Evaluate Zig-HS/D3 on the CDFv1 manifest format used in this repo.

Example
-------
python eval_d3_cdfv1.py \
    --cdf_root /media/tarun/B482367C823642E2/usr/cdfv1_onct_out \
    --cdf_csv /media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv \
    --d3_repo /path/to/D3 \
    --encoder_type ResNet-18 \
    --loss l2 \
    --num_frames 16 \
    --sampling first \
    --batch_size 16 \
    --num_workers 8

D3's raw second-order score is treated as a realness score in the original
repo. This script reports the usual fake-positive metrics using -raw_score.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


IMG_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

D3_ENCODERS = [
    "CLIP-16",
    "CLIP-32",
    "XCLIP-16",
    "XCLIP-32",
    "DINO-base",
    "DINO-large",
    "ResNet-18",
    "VGG-16",
    "EfficientNet-b4",
    "MobileNet-v3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate D3 on CDFv1.")
    parser.add_argument(
        "--cdf_root",
        default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out",
        help="CDFv1 root. Frames are read from <cdf_root>/<sample_dir>/image.png.",
    )
    parser.add_argument(
        "--cdf_csv",
        default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv",
        help="CDFv1 manifest CSV with sample_dir,label columns.",
    )
    parser.add_argument(
        "--d3_repo",
        default="",
        help="Path to cloned Zig-HS/D3 repo. Recommended for exact repo model.",
    )
    parser.add_argument("--encoder_type", default="ResNet-18", choices=D3_ENCODERS)
    parser.add_argument("--loss", default="l2", choices=["l2", "cos"])
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--sampling", default="first", choices=["first", "uniform"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--no_center_crop",
        action="store_true",
        help="Disable D3-style center crop by 10 percent of the long side.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use RGB order. Default mimics cv2.imread BGR order used by D3.",
    )
    parser.add_argument(
        "--save_results",
        default="d3_cdfv1_results.csv",
        help="Per-video CSV output path. Use empty string to disable.",
    )
    return parser.parse_args()


def video_id_from_sample_dir(sample_dir: str) -> str:
    sample_dir = str(sample_dir).replace("\\", "/")
    parts = Path(sample_dir).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])

    idx = basename.rfind("_frame_")
    if idx != -1:
        video_name = basename[:idx]
    else:
        match = re.search(r"_f\d+$", basename)
        video_name = basename[:match.start()] if match else basename

    return f"{prefix}/{video_name}" if prefix else video_name


def sample_indices(n_available: int, n_target: int, mode: str) -> np.ndarray:
    if n_available >= n_target:
        if mode == "first":
            return np.arange(n_target, dtype=int)
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def crop_center_by_percentage(img: Image.Image, percentage: float = 0.1) -> Image.Image:
    width, height = img.size
    if width > height:
        trim = int(width * percentage)
        return img.crop((trim, 0, width - trim, height))
    trim = int(height * percentage)
    return img.crop((0, trim, width, height - trim))


def load_frame(path: str, center_crop: bool, rgb: bool) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if center_crop:
        img = crop_center_by_percentage(img, 0.1)
    img = img.resize((IMG_SIZE, IMG_SIZE), RESAMPLE_BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if not rgb:
        arr = arr[:, :, ::-1].copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class CDFv1D3Dataset(Dataset):
    def __init__(
        self,
        cdf_root: str,
        cdf_csv: str,
        num_frames: int,
        sampling: str,
        center_crop: bool,
        rgb: bool,
    ):
        df = pd.read_csv(cdf_csv, sep=None, engine="python")
        required = {"sample_dir", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CDFv1 manifest must contain {required}. Found {list(df.columns)}")

        root = Path(cdf_root)
        df["label"] = df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)

        vid2paths = defaultdict(list)
        vid2labels = defaultdict(list)
        skipped = 0
        for _, row in df.iterrows():
            rel = str(row["sample_dir"]).replace("\\", "/")
            path = root / rel / "image.png"
            if path.is_file():
                vid2paths[row["video_id"]].append(str(path))
                vid2labels[row["video_id"]].append(int(row["label"]))
            else:
                skipped += 1

        self.videos = []
        self.mixed = 0
        for vid, paths in sorted(vid2paths.items()):
            labels = vid2labels[vid]
            if len(set(labels)) > 1:
                self.mixed += 1
            label = int(round(float(np.mean(labels))))
            self.videos.append((vid, sorted(paths), label))

        self.num_frames = num_frames
        self.sampling = sampling
        self.center_crop = center_crop
        self.rgb = rgb
        self.skipped = skipped

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = sample_indices(len(paths), self.num_frames, self.sampling)
        frames = []
        for frame_idx in indices:
            try:
                frames.append(load_frame(paths[int(frame_idx)], self.center_crop, self.rgb))
            except Exception:
                frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        return torch.stack(frames, dim=0), label, vid, len(paths)


def build_resnet18_fallback(loss_type: str):
    import torchvision.models as models

    class D3ResNet18Fallback(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                weights = models.ResNet18_Weights.DEFAULT
                resnet18 = models.resnet18(weights=weights)
            except Exception:
                resnet18 = models.resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet18.children())[:-1]).eval()

        def forward(self, x):
            bsz, time_steps, _, height, width = x.shape
            images = x.reshape(-1, 3, height, width)
            outputs = self.encoder(images).reshape(bsz, time_steps, -1)
            vec1 = outputs[:, :-1, :]
            vec2 = outputs[:, 1:, :]
            if loss_type == "cos":
                dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)
            else:
                dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)
            dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]
            return outputs, torch.mean(dis_2nd, dim=1), torch.std(dis_2nd, dim=1)

    return D3ResNet18Fallback()


def load_d3_model(args, device: torch.device):
    import_error = None
    if args.d3_repo:
        repo = Path(args.d3_repo).expanduser().resolve()
        sys.path.insert(0, str(repo / "models"))
        sys.path.insert(0, str(repo))

    try:
        from D3_model import D3_model

        model = D3_model(encoder_type=args.encoder_type, loss_type=args.loss)
        print("  D3 model source : imported from D3_model.py")
        print(f"  encoder_type    : {args.encoder_type}")
        return model.to(device).eval()
    except Exception as exc:
        import_error = exc

    if args.encoder_type != "ResNet-18":
        raise RuntimeError(
            f"Could not import D3_model.py ({import_error}). The embedded fallback "
            "only supports ResNet-18, so pass --d3_repo /path/to/D3 for "
            f"{args.encoder_type}."
        )

    print(f"  [INFO] Could not import D3_model.py ({import_error}).")
    print("  [INFO] Using embedded D3-compatible ResNet-18 fallback.")
    return build_resnet18_fallback(args.loss).to(device).eval()


def compute_metrics(labels: np.ndarray, fake_scores: np.ndarray):
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, fake_scores, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr)))
    youden_idx = int(np.nanargmax(tpr_arr - fpr_arr))
    threshold = float(thresholds[youden_idx])
    preds = (fake_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "auc": float(roc_auc_score(labels, fake_scores)),
        "ap": float(average_precision_score(labels, fake_scores)),
        "eer": float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0),
        "threshold": threshold,
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    sep = "=" * 88
    print("\n" + sep)
    print("D3 evaluation on CDFv1")
    print(sep)
    print(f"Device       : {device}")
    print(f"CDF root     : {args.cdf_root}")
    print(f"CDF CSV      : {args.cdf_csv}")
    print(f"D3 repo      : {args.d3_repo or '(not provided; fallback only for ResNet-18)'}")
    print(f"Encoder      : {args.encoder_type}")
    print(f"Loss         : {args.loss}")
    print(f"Frames/video : {args.num_frames} ({args.sampling})")
    print(f"Preprocess   : {IMG_SIZE}x{IMG_SIZE}, ImageNet norm, "
          f"{'no center crop' if args.no_center_crop else 'D3 center crop'}, "
          f"{'RGB' if args.rgb else 'BGR-like D3 channel order'}")

    dataset = CDFv1D3Dataset(
        cdf_root=args.cdf_root,
        cdf_csv=args.cdf_csv,
        num_frames=args.num_frames,
        sampling=args.sampling,
        center_crop=not args.no_center_crop,
        rgb=args.rgb,
    )
    if len(dataset) == 0:
        raise RuntimeError("No videos found. Check --cdf_root and --cdf_csv.")

    real_n = sum(1 for _, _, label in dataset.videos if label == 0)
    fake_n = sum(1 for _, _, label in dataset.videos if label == 1)
    frame_counts = np.array([len(paths) for _, paths, _ in dataset.videos])
    print("\nDataset")
    print(f"  Videos       : {len(dataset)}")
    print(f"  Real/Fake    : {real_n} / {fake_n}")
    print(f"  Frames/video : min={frame_counts.min()} median={np.median(frame_counts):.1f} "
          f"max={frame_counts.max()}")
    print(f"  Missing rows : {dataset.skipped}")
    if dataset.mixed:
        print(f"  [WARNING] Mixed-label grouped videos: {dataset.mixed}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print("\nLoading model")
    model = load_d3_model(args, device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not args.fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    video_ids, labels, raw_scores, avg_scores, n_frames_all = [], [], [], [], []
    print("\nRunning inference")
    with torch.inference_mode(), autocast_ctx:
        for frames, batch_labels, batch_vids, batch_frame_counts in tqdm(
            loader, desc=f"D3 {args.encoder_type}", unit="batch"
        ):
            frames = frames.to(device, non_blocking=True)
            _, dis_2nd_avg, dis_2nd_std = model(frames)
            raw_scores.extend(dis_2nd_std.float().cpu().numpy().tolist())
            avg_scores.extend(dis_2nd_avg.float().cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
            video_ids.extend(list(batch_vids))
            n_frames_all.extend(batch_frame_counts.numpy().tolist())

    labels_np = np.asarray(labels, dtype=int)
    raw_scores_np = np.asarray(raw_scores, dtype=np.float64)
    fake_scores_np = -raw_scores_np
    metrics = compute_metrics(labels_np, fake_scores_np)
    real_auc = roc_auc_score(1 - labels_np, raw_scores_np)
    real_ap = average_precision_score(1 - labels_np, raw_scores_np)

    print("\n" + sep)
    print("Results")
    print(sep)
    print("  Score convention")
    print("    raw D3 score        : second-order std, original realness score")
    print("    fake-positive score : -raw D3 score")
    print("\n  Fake-positive metrics")
    print(f"    AUC        : {metrics['auc']:.4f}")
    print(f"    AP         : {metrics['ap']:.4f}")
    print(f"    EER        : {metrics['eer'] * 100:.2f}%")
    print(f"    Best thr   : {metrics['threshold']:.6f}")
    print(f"    Acc        : {metrics['acc'] * 100:.2f}%")
    print(f"    F1         : {metrics['f1']:.4f}")
    print(f"    TPR/FPR    : {metrics['tpr'] * 100:.2f}% / {metrics['fpr'] * 100:.2f}%")
    print(
        f"    Confusion  : TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TN={metrics['tn']}"
    )
    print("\n  Raw real-positive orientation")
    print(f"    Real AUC   : {real_auc:.4f}")
    print(f"    Real AP    : {real_ap:.4f}")
    print("\n  Score stats")
    for label_value, name in [(0, "real"), (1, "fake")]:
        scores = raw_scores_np[labels_np == label_value]
        print(
            f"    {name:4s} raw mean={scores.mean():.6f} std={scores.std():.6f} "
            f"min={scores.min():.6f} max={scores.max():.6f}"
        )

    if args.save_results:
        out_path = Path(args.save_results)
        pd.DataFrame({
            "video_id": video_ids,
            "label": labels,
            "raw_d3_real_score": raw_scores,
            "fake_score": fake_scores_np,
            "dis_2nd_avg": avg_scores,
            "n_available_frames": n_frames_all,
            "encoder_type": args.encoder_type,
            "loss": args.loss,
        }).to_csv(out_path, index=False)
        print(f"\nSaved per-video results: {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
