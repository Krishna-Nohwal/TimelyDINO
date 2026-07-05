"""
cdfv1_test.py

Video-level evaluation on Celeb-DF v1 / CDFv1 using a Stage-2 VideoViT
checkpoint. CDFv1 is read from a manifest CSV with rows pointing to per-frame
directories:

    <cdf_root>/<sample_dir>/image.png

The manifest must contain:
    sample_dir, label

Labels follow the standard convention used in the Stage-2 training scripts:
    0 = real, 1 = fake

Example:
python cdfv1_test.py \
    --checkpoint /home/tarun/Desktop/best/best.pth \
    --cdf_root /media/tarun/B482367C823642E2/usr/cdfv1_onct_out \
    --cdf_csv /media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv \
    --num_frames 32 \
    --batch_size 4
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentations import load_and_resize, normalize
from cdfv2_knn42 import (
    IMG_SIZE,
    VideoViT,
    apply_real_bias,
    build_memory_bank_for_inference,
    compute_and_print_metrics,
    load_model,
    run_clip_inference,
    run_frame_inference,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a Stage-2 VideoViT checkpoint on CDFv1."
    )
    p.add_argument("--checkpoint", default="/home/tarun/Desktop/best/best.pth",
                   help="Path to Stage-2 frame-end .pth checkpoint.")
    p.add_argument("--cdf_root",
                   default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out",
                   help="CDFv1 preprocessed root.")
    p.add_argument("--cdf_csv",
                   default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv",
                   help="CDFv1 manifest CSV with columns sample_dir,label.")
    p.add_argument("--num_frames", default=32, type=int,
                   help="Frames sampled per video for temporal inference.")
    p.add_argument("--batch_size", default=4, type=int,
                   help="Number of videos per batch for temporal inference.")
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--topk", default=10, type=int,
                   help="k for top-k mean frame aggregation.")
    p.add_argument("--no_compile", action="store_true",
                   help="Skip torch.compile.")
    p.add_argument("--fp32", action="store_true",
                   help="Run inference in FP32 instead of FP16 autocast.")
    p.add_argument("--real_bias", default=0.0, type=float,
                   help="Power-transform bias applied to video scores < 0.5. "
                        "AUC is unaffected; threshold metrics change.")
    p.add_argument("--save_results", default="",
                   help="Optional per-video CSV output path.")
    p.add_argument("--train_real_root",
                   default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real",
                   help="Real training frames root, required only when checkpoint "
                        "uses the real-video memory bank.")
    p.add_argument("--knn_k", default=32, type=int)
    p.add_argument("--bank_batch_size", default=16, type=int)
    return p.parse_args()


def video_id_from_sample_dir(sample_dir: str) -> str:
    """
    Strip the trailing frame suffix from a manifest sample_dir.

    Examples:
      real/00011_f0052             -> real/00011
      fake/cdf/id1_id6_0007_f0072  -> fake/cdf/id1_id6_0007
      fake/abc_frame_31            -> fake/abc
    """
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


class CDFv1FrameDataset(Dataset):
    """Individual-frame CDFv1 loader for frame-level and video-mean metrics."""

    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        required = {"sample_dir", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CDFv1 manifest must contain {required}. Found: {list(df.columns)}")

        df["label"] = df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)

        root = Path(data_root)
        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            rel = str(row["sample_dir"]).replace("\\", "/")
            img_path = root / rel / "image.png"
            if img_path.is_file():
                self.samples.append((str(img_path), int(row["label"]), row["video_id"]))
            else:
                skipped += 1

        real_n = sum(1 for _, label, _ in self.samples if label == 0)
        fake_n = sum(1 for _, label, _ in self.samples if label == 1)
        print(f"  CDFv1 frames -> Real: {real_n} | Fake: {fake_n} | Total: {len(self.samples)}")
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing frame files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video_id = self.samples[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label, video_id


def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


class CDFv1ClipDataset(Dataset):
    """Video-level CDFv1 loader for temporal-transformer inference."""

    def __init__(self, csv_path: str, data_root: str, num_frames: int):
        self.num_frames = num_frames
        df = pd.read_csv(csv_path, sep=None, engine="python")
        required = {"sample_dir", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CDFv1 manifest must contain {required}. Found: {list(df.columns)}")

        df["label"] = df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)

        root = Path(data_root)
        vid2paths = defaultdict(list)
        vid2label = {}
        skipped = 0

        for _, row in df.iterrows():
            rel = str(row["sample_dir"]).replace("\\", "/")
            img_path = root / rel / "image.png"
            if img_path.is_file():
                vid = row["video_id"]
                vid2paths[vid].append(str(img_path))
                vid2label[vid] = int(row["label"])
            else:
                skipped += 1

        self.videos = []
        for vid, paths in sorted(vid2paths.items()):
            self.videos.append((vid, sorted(paths), vid2label[vid]))

        real_n = sum(1 for _, _, label in self.videos if label == 0)
        fake_n = sum(1 for _, _, label in self.videos if label == 1)
        print(f"  CDFv1 clips  -> Real: {real_n} | Fake: {fake_n} | Total: {len(self.videos)}")
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing frame files while grouping clips.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = _sample_frame_indices(len(paths), self.num_frames)
        frames = []
        for i in indices:
            try:
                img = load_and_resize(paths[i], IMG_SIZE)
                img = normalize(img)
            except Exception:
                img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            frames.append(img)
        return torch.stack(frames, dim=0), label, vid


def clip_collate_fn(batch):
    frames_list, labels, vids = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for frames in frames_list:
        pad_t = max_len - frames.size(0)
        if pad_t > 0:
            frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded.append(frames)
    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
        list(vids),
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    sep = "=" * 72
    print(f"\n  {sep}")
    print("  CDFv1 EVALUATION -- VideoViT Stage 2")
    print(f"  {sep}")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  CDF root    : {args.cdf_root}")
    print(f"  CDF CSV     : {args.cdf_csv}")
    print(f"  Num frames  : {args.num_frames}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Precision   : {'FP32' if args.fp32 else 'FP16 autocast'}")
    print(f"  Real bias   : {args.real_bias}")

    frame_dataset = CDFv1FrameDataset(args.cdf_csv, args.cdf_root)
    clip_dataset = CDFv1ClipDataset(args.cdf_csv, args.cdf_root, args.num_frames)

    persistent = args.num_workers > 0
    prefetch = 4 if args.num_workers > 0 else None
    frame_loader = DataLoader(
        frame_dataset,
        batch_size=args.batch_size * args.num_frames,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    clip_loader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    print(f"\n  Loading VideoViT (num_frames={args.num_frames}) ...")
    model, use_memory_bank = load_model(args.checkpoint, args.num_frames, device)
    if use_memory_bank:
        if not args.train_real_root:
            raise ValueError(
                "Checkpoint uses the real-video memory bank. Provide --train_real_root."
            )
        bank = build_memory_bank_for_inference(
            model=model,
            real_train_root=Path(args.train_real_root),
            num_frames=args.num_frames,
            k=args.knn_k,
            batch_size=args.bank_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        model.attach_memory_bank(bank)

    if not args.no_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile ...")
        model = torch.compile(model)

    print("\n  Running frame-level inference ...")
    (
        frame_labels,
        frame_probs,
        vid_ids_frame,
        vid_labels_frame,
        vid_mean_probs,
        vid_mean_logits,
        vid_topk_mean,
    ) = run_frame_inference(
        model, frame_loader, device, use_fp32=args.fp32, topk=args.topk
    )
    print(f"  Total frames evaluated: {len(frame_labels)}")
    print(f"  Total videos (frame aggregation): {len(vid_ids_frame)}")

    print("\n  Running clip-level temporal inference ...")
    vid_ids_clip, vid_labels_clip, vid_frame_end_probs, vid_no_frame_probs = run_clip_inference(
        model, clip_loader, device, use_fp32=args.fp32
    )
    print(f"  Total videos (temporal): {len(vid_ids_clip)}")

    bias_tag = ""
    if args.real_bias != 0.0:
        bias_tag = f" [real_bias={args.real_bias}]"
        vid_mean_probs = apply_real_bias(vid_mean_probs, args.real_bias)
        vid_mean_logits = apply_real_bias(vid_mean_logits, args.real_bias)
        vid_topk_mean = apply_real_bias(vid_topk_mean, args.real_bias)
        vid_frame_end_probs = apply_real_bias(vid_frame_end_probs, args.real_bias)
        vid_no_frame_probs = apply_real_bias(vid_no_frame_probs, args.real_bias)

    auc_frame = compute_and_print_metrics(
        frame_labels, frame_probs, f"(A) Frame-level{bias_tag} (CDFv1)"
    )
    auc_mean_probs = compute_and_print_metrics(
        vid_labels_frame, vid_mean_probs, f"(B) Video-mean probs{bias_tag} (CDFv1)"
    )
    auc_mean_logits = compute_and_print_metrics(
        vid_labels_frame, vid_mean_logits, f"(B) Video-mean logits->sigmoid{bias_tag} (CDFv1)"
    )
    auc_topk = compute_and_print_metrics(
        vid_labels_frame, vid_topk_mean, f"(B) Video-top{args.topk}-mean probs{bias_tag} (CDFv1)"
    )
    auc_frame_end = compute_and_print_metrics(
        vid_labels_clip, vid_frame_end_probs, f"(C) Video-temporal frame-end{bias_tag} (CDFv1)"
    )
    auc_no_frame = compute_and_print_metrics(
        vid_labels_clip, vid_no_frame_probs, f"(D) Video-temporal no-frame{bias_tag} (CDFv1)"
    )

    if args.save_results:
        frame_side = {
            vid: {
                "label": label,
                "mean_prob": mean_prob,
                "mean_logit_prob": mean_logit,
                f"top{args.topk}_mean_prob": topk_prob,
            }
            for vid, label, mean_prob, mean_logit, topk_prob in zip(
                vid_ids_frame,
                vid_labels_frame,
                vid_mean_probs,
                vid_mean_logits,
                vid_topk_mean,
            )
        }
        clip_frame_end = dict(zip(vid_ids_clip, vid_frame_end_probs))
        clip_no_frame = dict(zip(vid_ids_clip, vid_no_frame_probs))
        rows = []
        for vid in sorted(frame_side.keys()):
            row = {"video_id": vid}
            row.update(frame_side[vid])
            row["frame_end_prob"] = clip_frame_end.get(vid, float("nan"))
            row["no_frame_prob"] = clip_no_frame.get(vid, float("nan"))
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.save_results, index=False)
        print(f"\n  Per-video results written to: {args.save_results}")

    print(f"\n  {sep}")
    print(f"  FINAL SUMMARY [CDFv1]{bias_tag}")
    print(f"  {sep}")
    print(f"  (A) Frame-level AUC                    : {auc_frame:.4f}")
    print(f"  (B) Video-mean probs AUC               : {auc_mean_probs:.4f}")
    print(f"  (B) Video-mean logits->sigmoid AUC     : {auc_mean_logits:.4f}")
    print(f"  (B) Video-top{args.topk}-mean probs AUC      : {auc_topk:.4f}")
    print(f"  (C) Video-temporal frame-end AUC       : {auc_frame_end:.4f}  <- primary")
    print(f"  (D) Video-temporal no-frame AUC        : {auc_no_frame:.4f}")
    print(f"  {sep}")


if __name__ == "__main__":
    main()
