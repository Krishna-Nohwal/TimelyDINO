"""
cdfv3_test.py  —  Video-level AUC evaluation on Celeb-DF v3 face crops (CDFv3)
                   using a Stage 2 VideoViT checkpoint (video_model_knn.VideoViT).

Updated for the new Stage 1 backbone (frame_42.ViT) and to report BOTH
frame-level and video-temporal metrics (previously this script was frame-only
and used the standalone frame_model.ViT with no temporal transformer path).

Changes vs the original frame-only script
------------------------------------------
  - Loads the full VideoViT (video_model_knn.py) instead of a bare frame_model
    ViT, so the trained temporal transformers + fusion_classifier are used.
  - frame_42.ViT taps 5 layers [19,20,21,22,23]  (was 4 layers [20,21,22,23])
  - ViT.forward returns (logits_list, features_list, cls_list, fused_list)
    — a 4-tuple (was a 3-tuple)
  - Deepest frame head is now index 4 (was index 3); --mac_head_idx choices
    are now 0-4 and map to layers [19,20,21,22,23].
  - Adds a video-level CLIP dataset + clip_collate_fn + run_clip_inference,
    mirroring cdfv2_test_video.py, to produce stream (C) video-temporal AUC.
  - Adds --num_frames (frames sampled per video for the temporal pass, must
    match Stage 2 training) and --train_real_root / --knn_k / --bank_batch_size
    for checkpoints trained with a kNN memory bank.

Usage
-----
python cdfv3_test.py \
    --checkpoint  checkpoints_s2/best.pth \
    --cdfv3_root  /media/tarun/B482367C823642E2/usr/cdfv3_face_crops \
    [--cdfv3_csv  /path/to/manifest_cdfv3_face_crops.csv] \
    [--num_frames 32] [--batch_size 16] [--num_workers 4] \
    [--agg mean|max] [--topk 10] [--no_compile] [--save_results results.csv] \
    [--real_bias 0.0] [--mac_head_idx 4] [--fp32] \
    [--show_misclassified] [--misclassified_out misclassified.png] \
    [--max_display 64] [--misclassified_threshold 0.5] \
    [--train_real_root /path/to/real/training/frames] [--knn_k 32]

--show_misclassified
    After frame-level inference, collect all per-frame misclassified samples
    (FP + FN), render a grid of thumbnail images annotated with their true
    label, predicted probability, and video_id, and save it to
    --misclassified_out.

--misclassified_out  (default: misclassified.png)
    Path where the misclassified image grid is saved.

--max_display  (default: 64)
    Maximum number of misclassified frames to show in the grid (split evenly
    between FP and FN when possible).

--misclassified_threshold  (default: 0.5)
    Decision threshold used to classify a frame as Fake (prob >= threshold)
    or Real (prob < threshold).  Misclassification = ground-truth label
    disagrees with this decision.

Model
-----
Uses VideoViT built on frame_42.ViT (ViT-Large/16, EMBED_DIM=1024) with
SpatialHeads tapping layers [19,20,21,22,23], plus 5 temporal transformers
and a fusion_classifier (trained in Stage 2).  Frame-level inference (stream
A) uses --mac_head_idx (default 4, the deepest head, layer 23).  Video-level
temporal inference (stream C) uses the full VideoViT forward pass.

--real_bias  (float, default 0.0)
    Applies a downward pressure on scores < 0.5 (real-leaning predictions)
    at the VIDEO level only, to reduce false positive rate.

    Mechanism — power transform on the sub-0.5 region:
        p' = p^(1 + real_bias)   for p < 0.5
        p' = p                   for p >= 0.5

    • real_bias = 0.0  → identity (original behaviour, no change)
    • real_bias = 0.5  → mild suppression of real scores
    • real_bias = 1.0  → moderate suppression  (recommended starting point)
    • real_bias = 2.0  → aggressive suppression

    Because the transform is monotone and only acts below 0.5, AUC is
    preserved exactly.  Acc/F1/FPR/TPR at the 0.5 threshold DO change —
    that is the intended effect.

Directory layout expected
--------------------------
  <cdfv3_root>/
      manifest_cdfv3_face_crops.csv
      real/
          <video_id>/
              frame_000000/
                  image.png   (one crop per frame; multiple frames per video)
              frame_000031/
                  image.png
              ...
      fake/
          Celeb-synthesis/
              <video_id>/
                  frame_000000/
                      image.png
                  ...

  Manifest CSV columns required:
      sample_dir   — path to the per-frame directory relative to cdfv3_root
                     (e.g. "real/00011/frame_000000" or
                           "fake/Celeb-synthesis/id0_0000_test_.../frame_000000")
      label        — integer, 1 = Real, 0 = Fake  (as in the manifest)
      method       — acquisition method string (e.g. "real")
      video_stem   — numeric video id (e.g. 11 for video "00011")
      frame_idx    — frame index within the video (e.g. 0, 31, 62 …)

  The image file inside each sample_dir is expected to be named "image.png".

  video_id derivation:
      Taken from the immediate parent of sample_dir, which is the video folder.
      e.g.  "real/00011/frame_000000"  →  video_id = "00011"
            "fake/Celeb-synthesis/id0_0000_test_.../frame_000000"
                                        →  video_id = "id0_0000_test_..."

Video-level aggregation
-------------------------
  (B) Video-mean aggregation (no temporal transformer), three strategies:
        mean_probs   — mean of per-frame softmax probabilities
        mean_logits  — sigmoid( mean of per-frame fake-class logits )
        topk_mean    — mean of the top-k highest per-frame probabilities
  (C) Video-temporal — fused output of temporal transformers + fusion_classifier
        (the Stage 2 trained head; primary metric)

Outputs (all printed to stdout)
---------------------------------
  • Frame-level  : AUC, AP, Acc, F1, EER, TPR, FPR, TNR, confusion matrix
  • Video-level  : same metrics × 3 mean-aggregation strategies + temporal fusion
  • Optional CSV : one row per video with video_id, label, mean_prob,
                   mean_logit_prob, topk_mean_prob, temporal_prob
  • Optional PNG : grid of misclassified frame thumbnails (--show_misclassified)
"""

import os
import re
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score,
)

from augmentations import load_and_resize, normalize
from frame_model import ViT
from video_model import RealVideoMemoryBank, TemporalTransformer


# =========================================================================== #
# Frame-end Stage 2 model  (compatible with train_stage2_frame_end.py)
# =========================================================================== #

class VideoViT(torch.nn.Module):
    """
    Stage 2 frame-end video model:
      temporal_vec      : 4 temporal heads x 1024 = 4096 dims
      frame_mean_logits : deepest frame head averaged over valid frames = 2 dims

    This mirrors the model saved by train_stage2_frame_end.py. Memory, when
    enabled, is mixed into per-frame CLS sequences before temporal transformers.
    """

    EMBED_DIM = ViT.EMBED_DIM
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
        use_memory_bank: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_memory_bank = use_memory_bank
        self.memory_bank = None
        self.memory_gate = torch.nn.Parameter(
            torch.full((self.NUM_TEMPORAL_HEADS, 1, 1), -2.0)
        ) if use_memory_bank else None

        self.frame_model = ViT()
        self.temporal_transformers = torch.nn.ModuleList([
            TemporalTransformer(
                embed_dim=ViT.EMBED_DIM,
                num_frames=num_frames,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])
        self.fusion_classifier = torch.nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM + 2,
            2,
        )

    @property
    def vit(self):
        return self.frame_model.vit

    def attach_memory_bank(self, bank: RealVideoMemoryBank):
        if not self.use_memory_bank:
            raise RuntimeError("Model was not constructed with use_memory_bank=True.")
        if bank.num_heads != self.NUM_TEMPORAL_HEADS:
            raise ValueError(
                f"Bank has {bank.num_heads} heads but model expects "
                f"{self.NUM_TEMPORAL_HEADS}."
            )
        self.memory_bank = bank

    @staticmethod
    def _mean_valid_frame_logits(frame_logits_list, B, T, key_padding_mask, dtype):
        frame_logits = frame_logits_list[-1].float().reshape(B, T, 2)
        if key_padding_mask is None:
            return frame_logits.mean(dim=1).to(dtype=dtype)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        counts = valid.sum(dim=1).clamp(min=1)
        return ((frame_logits * valid).sum(dim=1) / counts).to(dtype=dtype)

    def forward(self, video: torch.Tensor, lengths=None):
        B, T, C, H, W = video.shape
        if T > self.num_frames:
            raise ValueError(f"Expected <= {self.num_frames} frames, got {T}")

        frames = video.reshape(B * T, C, H, W)
        frame_logits_list, frame_feats_list, cls_list = self.frame_model(frames)

        if lengths is None:
            key_padding_mask = None
        else:
            time_idx = torch.arange(T, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)

        cls_sequences = [
            cls_tokens.reshape(B, T, self.EMBED_DIM) for cls_tokens in cls_list
        ]

        if self.use_memory_bank:
            if self.memory_bank is None:
                raise RuntimeError(
                    "use_memory_bank=True but no bank attached. "
                    "Call attach_memory_bank() first."
                )
            memory_refs = self.memory_bank.query(cls_sequences, key_padding_mask)
        else:
            memory_refs = None

        video_feats_list = []
        for h, (temporal_tfm, frame_cls) in enumerate(
            zip(self.temporal_transformers, cls_sequences)
        ):
            if memory_refs is not None:
                gate = torch.sigmoid(self.memory_gate[h]).to(dtype=frame_cls.dtype)
                memory_ref = memory_refs[h].unsqueeze(1)
                frame_cls = (1 - gate) * frame_cls + gate * memory_ref
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)
        frame_mean_logits = self._mean_valid_frame_logits(
            frame_logits_list, B, T, key_padding_mask, temporal_vec.dtype
        )

        fused_with_frame = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        fused_no_frame = torch.cat(
            [temporal_vec, torch.zeros_like(frame_mean_logits)],
            dim=1,
        )

        video_logits_with_frame = self.fusion_classifier(fused_with_frame)
        video_logits_no_frame = self.fusion_classifier(fused_no_frame)

        return (
            video_logits_with_frame,
            video_logits_no_frame,
            frame_logits_list,
            frame_feats_list,
            video_feats_list,
        )


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a Stage 2 VideoViT checkpoint on CDFv3 face crops "
                    "(frame-level AND video-temporal AUC)."
    )
    p.add_argument("--checkpoint", default="frame_end_mem_gate0p13/best.pth",
                   help="Path to Stage 2 frame-end .pth checkpoint.")
    p.add_argument("--cdfv3_root",
                   default="/media/tarun/B482367C823642E2/usr/cdfv3_face_crops",
                   help="Root directory of the CDFv3 face-crops dataset "
                        "(must contain manifest_cdfv3_face_crops.csv, real/, fake/)")
    p.add_argument("--cdfv3_csv",
                   default="/media/tarun/B482367C823642E2/usr/cdfv3_face_crops/manifest_cdfv3_face_crops.csv",
                   help="Path to the CDFv3 manifest CSV. "
                        "Defaults to <cdfv3_root>/manifest_cdfv3_face_crops.csv "
                        "if not supplied.  Required columns: sample_dir, label "
                        "(optional: method, video_stem, frame_idx).")
    p.add_argument("--num_frames",    default=32,  type=int,
                   help="Frames sampled per video for temporal inference "
                        "(must match Stage 2 training). Default: 32.")
    p.add_argument("--batch_size",    default=16,  type=int,
                   help="Frame-level batch size (default 16 for ViT-Large VRAM budget). "
                        "Video-level (clip) batch size is computed as "
                        "max(1, batch_size // num_frames).")
    p.add_argument("--num_workers",   default=4,   type=int)
    p.add_argument("--agg",           default="mean", choices=["mean", "max"],
                   help="Legacy frame→video aggregation strategy (default: mean). "
                        "All three mean-based strategies (mean_probs, mean_logits, "
                        "topk_mean) are always evaluated; this flag only selects "
                        "which one is used for the optional --save_results CSV "
                        "'agg_prob' column.")
    p.add_argument("--topk",          default=10,  type=int,
                   help="k for top-k mean aggregation (default: 10). "
                        "Clipped to the number of frames if a video has fewer frames.")
    p.add_argument("--no_compile",    action="store_true",
                   help="Skip torch.compile (useful for debugging / older GPUs)")
    p.add_argument("--save_results",  default="",
                   help="If given, write a per-video CSV to this path")
    p.add_argument(
        "--real_bias", default=0.0, type=float,
        help=(
            "Downward bias applied to video-level scores < 0.5 (real-leaning) "
            "to reduce FPR.  Uses a power transform:  p' = p^(1+real_bias) for "
            "p < 0.5, identity for p >= 0.5.  "
            "0.0 = no change (default); 1.0 = moderate; 2.0 = aggressive.  "
            "AUC is unaffected (monotone transform); Acc/F1/FPR/TPR do change."
        ),
    )
    p.add_argument(
        "--mac_head_idx", default=3, type=int, choices=[0, 1, 2, 3],
        help=(
            "Which SpatialHead to use for frame-level inference.  Indices "
            "correspond to tapped layers [20, 21, 22, 23] respectively.  "
            "Default: 3 (layer 23, deepest)."
        ),
    )
    p.add_argument(
        "--fp32", action="store_true",
        help="Disable autocast and run inference in full FP32. Slower but useful "
             "for debugging numerical issues on ViT-Large."
    )

    # ── kNN memory bank (only needed if checkpoint was trained with one) ────
    p.add_argument("--train_real_root", default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real", type=str,
                   help="Root dir of real training frames (required when checkpoint "
                        "uses use_memory_bank=True, to rebuild the real-video kNN "
                        "bank). Layout: <train_real_root>/<video_id>/<frame_dir>/image.png "
                        "(same per-frame-directory structure as CDFv3).")
    p.add_argument("--knn_k",           default=32,  type=int,
                   help="Number of nearest real neighbours for kNN bank (default: 32).")
    p.add_argument("--bank_batch_size", default=16,  type=int,
                   help="Batch size (videos) for building the kNN memory bank (default: 16).")

    # ── Misclassified display ────────────────────────────────────────────────
    p.add_argument(
        "--show_misclassified", action="store_true",
        help="Save a grid of misclassified frame thumbnails to --misclassified_out."
    )
    p.add_argument(
        "--misclassified_out", default="misclassified.png",
        help="Output path for the misclassified image grid (default: misclassified.png)."
    )
    p.add_argument(
        "--max_display", default=64, type=int,
        help="Maximum number of misclassified frames to show (split evenly between "
             "FP and FN when possible).  Default: 64."
    )
    p.add_argument(
        "--misclassified_threshold", default=0.5, type=float,
        help="Decision threshold for labelling a frame Fake (prob >= threshold). "
             "Default: 0.5."
    )

    return p.parse_args()


# =========================================================================== #
# Constants
# =========================================================================== #

IMG_SIZE = 256   # must match training (16×16 patch grid: 256 // 16 = 16 patches per side)

# frame_model.ViT taps 4 layers; SpatialHead index -> tapped transformer layer.
_TAPPED_LAYERS = [20, 21, 22, 23]


def video_id_from_sample_dir(sample_dir: str) -> str:
    """
    Derive a video_id from a sample_dir column entry.

    sample_dir points to the per-frame directory (e.g. "real/00011/frame_000000").
    The video_id is the name of its *parent* directory, which groups all frames
    belonging to the same video.

    Examples (sample_dir relative to cdfv3_root):
        "real/00011/frame_000000"
            →  "00011"
        "real/00021/frame_000030"
            →  "00021"
        "fake/Celeb-synthesis/id0_0000_test_id01822_wV0Fl0ZN7Vg/frame_000000"
            →  "id0_0000_test_id01822_wV0Fl0ZN7Vg"
    """
    return Path(sample_dir).parent.name


# =========================================================================== #
# Frame dataset  (stream A — single-frame inference via frame_model)
# =========================================================================== #

class CDFv3Dataset(Dataset):
    """
    Loads individual face-crop frames from the CDFv3 manifest CSV.

    Each __getitem__ returns (image_tensor, label, video_id, img_path) so
    that the evaluation loop can group frames into videos and retrieve raw
    images for misclassification visualisation.

    Manifest CSV columns used:
        sample_dir   — path to the per-frame *directory* relative to cdfv3_root
                       (e.g. "real/00011/frame_000000").  The actual image is
                       expected at  <cdfv3_root>/<sample_dir>/image.png
        label        — integer label as stored in the manifest
                       (1 = Real, 0 = Fake — note: inverted from the old schema;
                        the dataset prints counts accordingly)

    video_id is derived as the parent directory of sample_dir, which is the
    per-video folder grouping all frame sub-directories.
    """

    FRAME_IMAGE_NAME = "image.png"

    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")

        required = {"sample_dir", "label"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Manifest CSV is missing required columns: {missing_cols}\n"
                f"Found columns: {list(df.columns)}"
            )

        df["label"] = df["label"].astype(int)

        # Manifest convention: label=1 → Real, label=0 → Fake
        real_n = (df["label"] == 1).sum()
        fake_n = (df["label"] == 0).sum()
        print(f"  CDFv3 manifest  →  Real: {real_n}  |  Fake: {fake_n}  |  Total: {len(df)}"
              f"  [manifest: 1=Real / 0=Fake]")

        root = Path(data_root)

        img_paths = df["sample_dir"].apply(
            lambda d: str(root / d / self.FRAME_IMAGE_NAME)
        )
        labels    = df["label"].values
        video_ids = df["sample_dir"].apply(video_id_from_sample_dir).values

        exists_mask = np.array([os.path.exists(p) for p in img_paths])
        n_skip = int((~exists_mask).sum())
        if n_skip:
            print(f"  [CDFv3] Skipped {n_skip} missing image files "
                  f"({exists_mask.sum()} remaining)")

        self.entries = list(zip(
            np.array(img_paths)[exists_mask],
            labels[exists_mask],
            video_ids[exists_mask],
        ))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label, video_id = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label, video_id, img_path


# =========================================================================== #
# Clip dataset  (stream C — full VideoViT temporal inference)
# =========================================================================== #

def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    """Uniform stride; tiles if fewer frames than needed."""
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


class CDFv3ClipDataset(Dataset):
    """
    Groups per-frame directories (from the manifest) into videos, samples
    num_frames per video with uniform stride.  Used for the temporal
    transformer forward pass (stream C).

    Built from the same manifest CSV as CDFv3Dataset so video grouping and
    label convention (1=Real, 0=Fake in the manifest) stay consistent; labels
    are remapped to the standard 0=Real/1=Fake convention here.
    """

    FRAME_IMAGE_NAME = "image.png"

    def __init__(self, csv_path: str, data_root: str, num_frames: int):
        self.num_frames = num_frames

        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"]    = df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)

        root = Path(data_root)
        vid2paths: dict = defaultdict(list)
        vid2label: dict = {}

        for video_id, group in df.groupby("video_id"):
            # Manifest: 1=Real, 0=Fake -> remap to standard 0=Real, 1=Fake.
            manifest_label = int(group["label"].iloc[0])
            label = 0 if manifest_label == 1 else 1

            paths = []
            for rel in group["sample_dir"]:
                img_path = root / rel / self.FRAME_IMAGE_NAME
                if img_path.is_file():
                    paths.append(str(img_path))
            if not paths:
                continue
            vid2paths[video_id] = sorted(paths)
            vid2label[video_id] = label

        self.videos = [
            (vid, paths, vid2label[vid]) for vid, paths in sorted(vid2paths.items())
        ]

        real_n = sum(1 for _, _, l in self.videos if l == 0)
        fake_n = sum(1 for _, _, l in self.videos if l == 1)
        print(f"  CDFv3 clips   →  Real: {real_n}  |  Fake: {fake_n}  |  "
              f"Total: {len(self.videos)}  [standard: 0=Real / 1=Fake]")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = _sample_frame_indices(len(paths), self.num_frames)
        frames  = []
        for i in indices:
            try:
                img = load_and_resize(paths[i], IMG_SIZE)
                img = normalize(img)
            except Exception:
                img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            frames.append(img)
        return torch.stack(frames, dim=0), label, vid   # (T, 3, H, W)


def clip_collate_fn(batch):
    """Pads variable-length clips; returns (frames, labels, lengths, video_ids)."""
    frames_list, labels, vids = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded  = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        padded.append(F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t)) if pad_t > 0 else f)
    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
        list(vids),
    )


# =========================================================================== #
# Metrics helper
# =========================================================================== #

def compute_and_print_metrics(labels, probs, level: str) -> float:
    """
    Compute AUC, AP, Acc, F1, EER, TPR/FPR/TNR and confusion matrix.
    Returns AUC.
    """
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)

    fpr_arr, tpr_arr, _ = roc_curve(labels, probs, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer     = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sep = "─" * 72
    print(f"\n  {sep}")
    print(f"  [{level}]")
    print(f"  {sep}")
    print(f"  AUC  : {auc:.4f}")
    print(f"  AP   : {ap:.4f}")
    print(f"  Acc  : {acc * 100:.2f}%")
    print(f"  F1   : {f1:.4f}")
    print(f"  EER  : {eer * 100:.2f}%")
    print(f"  TPR  : {tpr * 100:.2f}%   FPR : {fpr * 100:.2f}%   TNR : {tnr * 100:.2f}%")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  {sep}")

    return float(auc)


# =========================================================================== #
# Real-score bias (video-level only)
# =========================================================================== #

def apply_real_bias(probs: list, real_bias: float) -> list:
    """
    Suppress scores below 0.5 using a power transform, leaving scores >= 0.5
    completely untouched.  AUC is preserved (monotone transform).

        p' = p ^ (1 + real_bias)   for p < 0.5
        p' = p                     for p >= 0.5
    """
    if real_bias == 0.0:
        return probs

    exponent = 1.0 + real_bias
    return [p ** exponent if p < 0.5 else p for p in probs]


# =========================================================================== #
# Inference  --  stream (A) frame-level  +  (B) video-mean aggregation
# =========================================================================== #

def run_frame_inference(model: VideoViT, loader: DataLoader, device: torch.device,
                        mac_head_idx: int = 3, use_fp32: bool = False):
    """
    Single-frame forward pass through the frozen frame_model only.

    Returns five parallel lists:
        frame_labels   — int   (0=Real / 1=Fake, manifest convention already
                                 inverted by the caller before this is used in
                                 metrics)
        frame_probs    — float (P(fake) from softmax)
        frame_logits   — float (raw fake-class logit, pre-softmax)
        frame_vids     — str   (video_id string)
        frame_paths    — str   (absolute path to image file)

    mac_head_idx : int
        Index into logits_list / features_list (0-3).
        Maps to frame_model.ViT tapped layers [20, 21, 22, 23].
        Default 3 -> layer 23 (deepest, highest-level semantics).
    """
    assert 0 <= mac_head_idx <= 3, f"mac_head_idx must be 0-3, got {mac_head_idx}"
    print(f"  Using SpatialHead index {mac_head_idx} "
          f"(tapped layer {_TAPPED_LAYERS[mac_head_idx]})")

    frame_labels: list = []
    frame_probs:  list = []
    frame_logits: list = []
    frame_vids:   list = []
    frame_paths:  list = []

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not use_fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Access the frozen frame_model directly -- no temporal transformer needed here.
    frame_model = (
        model._orig_mod.frame_model
        if hasattr(model, "_orig_mod")
        else model.frame_model
    )
    frame_model.eval()

    with torch.inference_mode(), autocast_ctx:
        for imgs, labels, video_ids, img_paths in tqdm(loader, desc="Frame inference", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)

            # frame_model.ViT returns (logits_list, features_list, cls_list)
            logits_list, _, _ = frame_model(imgs)

            raw_logits = logits_list[mac_head_idx].float()   # (B, 2)
            probs      = torch.softmax(raw_logits, dim=1)[:, 1].cpu().numpy()
            fake_logit = raw_logits[:, 1].cpu().numpy()

            frame_probs.extend(probs.tolist())
            frame_logits.extend(fake_logit.tolist())
            frame_labels.extend(labels.numpy().tolist())
            frame_vids.extend(list(video_ids))
            frame_paths.extend(list(img_paths))

    return frame_labels, frame_probs, frame_logits, frame_vids, frame_paths


# =========================================================================== #
# Inference  --  stream (C) video-temporal via clip loader
# =========================================================================== #

def run_clip_inference(model: VideoViT, loader: DataLoader, device: torch.device,
                       use_fp32: bool = False):
    """
    Full VideoViT forward pass through temporal transformers + fusion_classifier.
    Returns (video_ids, video_labels, video_frame_end_probs, video_no_frame_probs).

    Labels here follow the STANDARD convention (0=Real, 1=Fake) since
    CDFv3ClipDataset already remaps from the manifest's 1=Real/0=Fake.
    """
    video_ids_out:    list = []
    video_labels_out: list = []
    video_frame_end_probs: list = []
    video_no_frame_probs: list = []

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not use_fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    model.eval()

    with torch.inference_mode(), autocast_ctx:
        for frames, labels, lengths, vids in tqdm(
            loader, desc="Clip inference", unit="batch"
        ):
            frames  = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            video_logits_with_frame, video_logits_no_frame, _, _, _ = model(frames, lengths)
            probs_with_frame = torch.softmax(
                video_logits_with_frame.float(), dim=1
            )[:, 1].cpu().numpy()
            probs_no_frame = torch.softmax(
                video_logits_no_frame.float(), dim=1
            )[:, 1].cpu().numpy()

            video_frame_end_probs.extend(probs_with_frame.tolist())
            video_no_frame_probs.extend(probs_no_frame.tolist())
            video_labels_out.extend(labels.numpy().tolist())
            video_ids_out.extend(vids)

    return video_ids_out, video_labels_out, video_frame_end_probs, video_no_frame_probs


# =========================================================================== #
# Misclassified image grid
# =========================================================================== #

def display_misclassified(
    frame_labels: list,
    frame_probs:  list,
    frame_vids:   list,
    frame_paths:  list,
    out_path:     str,
    threshold:    float = 0.5,
    max_display:  int   = 64,
    thumb_size:   int   = 160,
) -> None:
    """
    Collects misclassified frames, renders a labelled grid, and saves it.

    Layout
    ------
    Two sections side-by-side:

      ┌─────────────────────────────┬─────────────────────────────┐
      │  FALSE POSITIVES            │  FALSE NEGATIVES            │
      │  (Real predicted as Fake)   │  (Fake predicted as Real)   │
      │  border: orange             │  border: red                │
      └─────────────────────────────┴─────────────────────────────┘

    Each thumbnail is annotated with:
      • top-left corner: true label (R/F)
      • bottom bar:      P(fake) score
      • short video_id

    Parameters
    ----------
    frame_labels : list[int]   ground-truth (0=Real, 1=Fake)
    frame_probs  : list[float] P(fake) per frame
    frame_vids   : list[str]   video_id per frame
    frame_paths  : list[str]   absolute image paths
    out_path     : str         where to save the PNG
    threshold    : float       decision boundary (default 0.5)
    max_display  : int         cap on total images shown
    thumb_size   : int         pixels per thumbnail side
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
    except ImportError as e:
        print(f"\n  [WARNING] Cannot display misclassified images: {e}")
        print("  Install matplotlib and Pillow:  pip install matplotlib Pillow")
        return

    # ── Collect FP and FN ────────────────────────────────────────────────────
    fp_items: list = []
    fn_items: list = []

    for lbl, prob, vid, path in zip(frame_labels, frame_probs, frame_vids, frame_paths):
        pred = 1 if prob >= threshold else 0
        if pred != lbl:
            entry = {"path": path, "prob": prob, "vid": vid, "label": lbl}
            if lbl == 0:
                fp_items.append(entry)   # Real → predicted Fake
            else:
                fn_items.append(entry)   # Fake → predicted Real

    total_fp = len(fp_items)
    total_fn = len(fn_items)
    print(f"\n  Misclassified frames — FP (Real→Fake): {total_fp} | "
          f"FN (Fake→Real): {total_fn} | "
          f"Total: {total_fp + total_fn}")

    if total_fp + total_fn == 0:
        print("  No misclassified frames — grid not generated.")
        return

    # Sort by most confident wrong prediction (highest prob for FP, lowest for FN)
    fp_items.sort(key=lambda x: x["prob"], reverse=True)
    fn_items.sort(key=lambda x: x["prob"])

    # Cap total to max_display, splitting evenly
    half = max_display // 2
    fp_show = fp_items[:min(half, len(fp_items))]
    fn_show = fn_items[:min(half, len(fn_items))]

    # If one side has fewer, give the slack to the other
    slack_fp = half - len(fp_show)
    slack_fn = half - len(fn_show)
    if slack_fp > 0 and len(fn_items) > len(fn_show):
        fn_show = fn_items[:min(half + slack_fp, len(fn_items))]
    if slack_fn > 0 and len(fp_items) > len(fp_show):
        fp_show = fp_items[:min(half + slack_fn, len(fp_items))]

    # ── Grid layout ─────────────────────────────────────────────────────────
    cols = 8                               # thumbnails per row within each section
    rows_fp = math.ceil(len(fp_show) / cols)
    rows_fn = math.ceil(len(fn_show) / cols)
    max_rows = max(rows_fp, rows_fn, 1)

    PAD        = 6     # px border around each thumb
    LABEL_H    = 22    # px text bar at bottom of each thumb
    HEADER_H   = 50    # section title height in pixels
    SECTION_W  = cols * (thumb_size + PAD * 2)
    FIG_W_PX   = SECTION_W * 2 + 60       # 60px gutter
    FIG_H_PX   = HEADER_H + max_rows * (thumb_size + PAD * 2 + LABEL_H) + 40

    DPI   = 100
    fig_w = FIG_W_PX / DPI
    fig_h = FIG_H_PX / DPI

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        f"Misclassified Frames  |  threshold={threshold:.2f}  |  "
        f"FP={total_fp}  FN={total_fn}  (showing {len(fp_show)}+{len(fn_show)})",
        color="white", fontsize=11, fontweight="bold", y=0.98,
    )

    BORDER_FP = "#ff9500"   # orange  — Real labelled as Fake
    BORDER_FN = "#ff3b30"   # red     — Fake labelled as Real
    BG_COLOR  = "#1a1a2e"

    def _render_section(items, section_col, title, border_color):
        """Draw a grid of thumbs in one half of the figure."""
        ax_title = fig.add_axes([
            section_col * 0.5 + 0.01,
            1.0 - (HEADER_H / FIG_H_PX) - 0.01,
            0.48,
            HEADER_H / FIG_H_PX,
        ])
        ax_title.set_facecolor(BG_COLOR)
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5, title,
            ha="center", va="center",
            color=border_color, fontsize=10, fontweight="bold",
            transform=ax_title.transAxes,
        )

        for i, item in enumerate(items):
            row = i // cols
            col = i %  cols

            cell_w   = 1.0 / (cols * 2 + 1)
            cell_h   = (1.0 - HEADER_H / FIG_H_PX - 0.06) / max_rows
            x0 = (section_col * cols + col) / (cols * 2) * 0.98 + 0.01
            y0 = 1.0 - HEADER_H / FIG_H_PX - 0.02 - (row + 1) * cell_h

            ax = fig.add_axes([x0, y0, cell_w * 1.7, cell_h * 0.88])
            ax.set_facecolor(BG_COLOR)

            try:
                pil_img = Image.open(item["path"]).convert("RGB")
                pil_img = pil_img.resize((thumb_size, thumb_size), Image.BILINEAR)
                ax.imshow(np.array(pil_img))
            except Exception:
                ax.set_facecolor("#333")

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            ax.set_xticks([])
            ax.set_yticks([])

            lbl_str = "R" if item["label"] == 0 else "F"
            ax.text(
                0.04, 0.96, lbl_str,
                transform=ax.transAxes,
                color="white", fontsize=7, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#007aff" if item["label"] == 0 else "#ff2d55",
                          alpha=0.85, linewidth=0),
            )

            ax.text(
                0.5, 0.02,
                f"p={item['prob']:.3f}",
                transform=ax.transAxes,
                color="white", fontsize=6, va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="black", alpha=0.65, linewidth=0),
            )

            vid_short = item["vid"].split("/")[-1][-18:]
            ax.set_title(vid_short, color="#aaaaaa", fontsize=5, pad=2)

    _render_section(fp_show, 0,
                    f"FALSE POSITIVES — Real predicted as Fake  ({len(fp_show)}/{total_fp})",
                    BORDER_FP)
    _render_section(fn_show, 1,
                    f"FALSE NEGATIVES — Fake predicted as Real  ({len(fn_show)}/{total_fn})",
                    BORDER_FN)

    fig.add_artist(plt.Line2D(
        [0.5, 0.5], [0.02, 0.96],
        transform=fig.transFigure,
        color="#444", linewidth=1.5, linestyle="--",
    ))

    legend_elements = [
        mpatches.Patch(facecolor=BORDER_FP, label="FP (Real→Fake)"),
        mpatches.Patch(facecolor=BORDER_FN, label="FN (Fake→Real)"),
        mpatches.Patch(facecolor="#007aff", label="True=Real (R)"),
        mpatches.Patch(facecolor="#ff2d55", label="True=Fake (F)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=4,
        framealpha=0.3, facecolor="#1a1a2e",
        labelcolor="white", fontsize=7,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Misclassified image grid saved → {out_path}")


# =========================================================================== #
# Video-level aggregation  (three mean-based strategies, stream B)
# =========================================================================== #

def aggregate_to_video(frame_labels, frame_probs, frame_logits, frame_vids,
                       agg: str = "mean", topk: int = 10):
    """
    Groups per-frame predictions by video_id and computes three aggregations:

      mean_probs   — mean of per-frame softmax probabilities       (original)
      mean_logits  — sigmoid( mean of per-frame fake-class logits )
      topk_mean    — mean of the top-k highest per-frame probs

    Returns:
        video_ids        — list[str]
        video_labels     — list[int]
        vid_mean_probs   — list[float]
        vid_mean_logits  — list[float]
        vid_topk_mean    — list[float]
    """
    vid2labels:  dict = defaultdict(list)
    vid2probs:   dict = defaultdict(list)
    vid2logits:  dict = defaultdict(list)

    for lbl, prob, logit, vid in zip(frame_labels, frame_probs,
                                     frame_logits, frame_vids):
        vid2labels[vid].append(lbl)
        vid2probs[vid].append(prob)
        vid2logits[vid].append(logit)

    video_ids        = sorted(vid2labels.keys())
    video_labels     = []
    vid_mean_probs   = []
    vid_mean_logits  = []
    vid_topk_mean    = []

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    for vid in video_ids:
        lbls   = vid2labels[vid]
        unique = set(lbls)
        if len(unique) > 1:
            print(f"  [WARNING] Video '{vid}' has mixed labels {unique}; "
                  "using majority vote.")
        video_labels.append(int(round(np.mean(lbls))))

        frame_p = np.array(vid2probs[vid])
        frame_l = np.array(vid2logits[vid])

        vid_mean_probs.append(float(frame_p.mean()))
        vid_mean_logits.append(float(_sigmoid(frame_l.mean())))

        k = min(topk, len(frame_p))
        topk_probs = np.partition(frame_p, -k)[-k:]
        vid_topk_mean.append(float(topk_probs.mean()))

    return (video_ids, video_labels,
            vid_mean_probs, vid_mean_logits, vid_topk_mean)


# =========================================================================== #
# Training real-video dataset  (for kNN bank construction at inference time)
# =========================================================================== #

class TrainRealClipDataset(Dataset):
    """
    Loads only real training video clips for kNN memory bank construction.

    Supports two directory layouts, auto-detected:

    (a) Nested  — <real_train_root>/<video_id>/<frame_dir>/image.png
        (the layout used by CDFv3). Each immediate subdirectory IS a video;
        every "image.png" found anywhere beneath it is one of that video's
        frames.

    (b) Flat    — <real_train_root>/<video_id>_frame_NN/image.png  or
                  <real_train_root>/<video_id>_fNNNN/image.png
        (the layout used by some FF++ preprocessing pipelines, e.g.
        onct_preprocessed_out/real). Each immediate subdirectory is a
        single FRAME, not a video — frames must be grouped by stripping
        the trailing _frame_NN / _fNNNN suffix to recover the video_id.
        Without this grouping every frame is miscounted as its own video
        (e.g. 1,000 real videos x 32 frames -> reported as 32,000 "videos").

    Args
    ----
    real_train_root : Path — root dir of real training videos.
    num_frames      : int  — frames to sample per video (must match training).
    """

    def __init__(self, real_train_root: Path, num_frames: int):
        self.num_frames = num_frames
        self.videos: list = []

        # Detect layout: if any immediate subdirectory's name ends in a
        # frame suffix (_frame_NN or _fNNNN), treat the whole root as FLAT
        # and group by stripped video_id. Otherwise treat as NESTED.
        subdirs = sorted(d for d in real_train_root.iterdir() if d.is_dir())
        is_flat = any(re.search(r'_(?:frame_|f)\d+$', d.name) for d in subdirs)

        if is_flat:
            vid2paths: dict = defaultdict(list)
            for d in subdirs:
                img_path = d / "image.png"
                if not img_path.exists():
                    continue
                vid = re.sub(r'_(?:frame_|f)\d+$', '', d.name)
                vid2paths[vid].append(str(img_path))
            for vid, paths in sorted(vid2paths.items()):
                self.videos.append((vid, sorted(paths)))
        else:
            for video_dir in subdirs:
                paths = sorted(str(p) for p in video_dir.rglob("image.png"))
                if paths:
                    self.videos.append((video_dir.name, paths))

        layout_tag = "flat (grouped by stripped video_id)" if is_flat else "nested (<video_id>/<frame_dir>)"
        print(f"  [TrainRealClipDataset] layout detected: {layout_tag}")
        print(f"  [TrainRealClipDataset] {len(self.videos)} real training videos found.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths  = self.videos[idx]
        indices     = _sample_frame_indices(len(paths), self.num_frames)
        frames      = []
        for i in indices:
            try:
                img = load_and_resize(paths[i], IMG_SIZE)
                img = normalize(img)
            except Exception:
                img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            frames.append(img)
        return torch.stack(frames, dim=0), 0   # label=0 always, needed for collate


def _real_clip_collate_fn(batch):
    """Collate for TrainRealClipDataset — pads to max clip length."""
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded  = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        padded.append(F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t)) if pad_t > 0 else f)
    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
    )


def build_memory_bank_for_inference(
    model:           "VideoViT",
    real_train_root: Path,
    num_frames:      int,
    k:               int,
    batch_size:      int,
    num_workers:     int,
    device:          torch.device,
) -> "RealVideoMemoryBank":
    """
    Rebuild the real-video kNN memory bank from training data for inference.

    Mirrors build_memory_bank() from train_stage2.py — calls frame_model and
    temporal_transformers directly (not the full forward pass) to avoid the
    chicken-and-egg problem of querying a bank that doesn't exist yet.

    The bank is attached to the model after building via model.attach_memory_bank().
    """
    print("\n  Building real-video kNN memory bank from training data …")

    dataset = TrainRealClipDataset(real_train_root, num_frames=num_frames)
    if len(dataset) == 0:
        raise RuntimeError(
            "No real training videos found. Check --train_real_root."
        )

    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        collate_fn         = _real_clip_collate_fn,
        persistent_workers = num_workers > 0,
        prefetch_factor    = 4 if num_workers > 0 else None,
    )

    bank = RealVideoMemoryBank(
        embed_dim = VideoViT.EMBED_DIM,
        num_heads = VideoViT.NUM_TEMPORAL_HEADS,
        k         = k,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    was_training = model.training
    model.eval()

    with torch.inference_mode(), \
         torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, _, lengths in tqdm(loader, desc="  Building bank", leave=False):
            B, T, C, H, W = frames.shape
            frames  = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            flat_frames = frames.reshape(B * T, C, H, W)

            # frame_model.ViT returns (logits_list, features_list, cls_list)
            _, _, cls_list = raw_model.frame_model(flat_frames)

            time_idx         = torch.arange(T, device=device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.unsqueeze(1)

            cls_sequences = []
            for cls_tokens in cls_list:
                frame_cls = cls_tokens.reshape(B, T, raw_model.EMBED_DIM)
                cls_sequences.append(frame_cls)

            bank.add(cls_sequences, key_padding_mask)

    bank.build()

    if was_training:
        model.train()

    print(f"  Memory bank ready: {len(bank)} real-video CLS prototypes, k={k}")
    return bank


# =========================================================================== #
# Model loading helper  (auto-detects use_memory_bank from checkpoint shape)
# =========================================================================== #

def load_model(checkpoint_path: str, num_frames: int, device: torch.device) -> VideoViT:
    """
    Load a VideoViT checkpoint, auto-detecting use_memory_bank from the
    fusion_classifier weight shape so the model is always instantiated with
    the same architecture that was saved.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}

    fc1_key = "frame_model.spatial_heads.0.head.0.weight"
    if fc1_key in ckpt:
        expected = (1024, 3072)
        actual = tuple(ckpt[fc1_key].shape)
        if actual != expected:
            raise ValueError(
                f"Checkpoint appears to be from a different model size.\n"
                f"  Expected {fc1_key} shape : {expected}  (ViT-Large)\n"
                f"  Got                      : {actual}"
            )
        print(f"  frame_model SpatialHead shape check passed {actual}")
    else:
        print(f"  [WARNING] Could not find '{fc1_key}' in checkpoint for shape check.")

    fusion_key = "fusion_classifier.weight"
    if fusion_key not in ckpt:
        raise KeyError(
            f"Cannot find '{fusion_key}' in checkpoint -- is this a VideoViT checkpoint?"
        )
    fusion_in_dim = ckpt[fusion_key].shape[1]
    expected_fusion_dim = VideoViT.NUM_TEMPORAL_HEADS * VideoViT.EMBED_DIM + 2
    if fusion_in_dim != expected_fusion_dim:
        raise ValueError(
            f"Unexpected fusion_classifier input dim {fusion_in_dim}. "
            f"Expected {expected_fusion_dim} for frame-end model "
            f"({VideoViT.NUM_TEMPORAL_HEADS}*{VideoViT.EMBED_DIM}+2)."
        )

    use_memory_bank = "memory_gate" in ckpt
    print(f"  fusion_classifier input dim={fusion_in_dim} -> frame-end model")
    print(f"  memory_gate present={use_memory_bank} -> use_memory_bank={use_memory_bank}")

    model = VideoViT(
        num_frames=num_frames,
        use_memory_bank=use_memory_bank,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("  Checkpoint loaded successfully.")
    return model, use_memory_bank

    # Shape check on frame_model SpatialHead head[0] (frame_42.SpatialHead.head)
    # head : nn.Sequential(Linear(2*embed_dim, embed_dim//2), ReLU, Dropout)
    # so head.0.weight has shape (embed_dim//2, 2*embed_dim) = (512, 2048)
    fc1_key = "frame_model.spatial_heads.0.head.0.weight"
    if fc1_key in ckpt:
        expected = (1024, 3072)
        actual   = tuple(ckpt[fc1_key].shape)
        if actual != expected:
            raise ValueError(
                f"Checkpoint appears to be from a different model size.\n"
                f"  Expected {fc1_key} shape : {expected}  (ViT-Large)\n"
                f"  Got                      : {actual}"
            )
        print(f"  ✓ frame_model SpatialHead shape check passed {actual}")
    else:
        print(f"  [WARNING] Could not find '{fc1_key}' in checkpoint for shape check.")

    # Auto-detect use_memory_bank from fusion_classifier input dim:
    #   5122 = NUM_TEMPORAL_HEADS*EMBED_DIM + 2                      -> False
    #   5127 = NUM_TEMPORAL_HEADS*EMBED_DIM + 2 + NUM_TEMPORAL_HEADS -> True
    fusion_key = "fusion_classifier.weight"
    if fusion_key not in ckpt:
        raise KeyError(
            f"Cannot find '{fusion_key}' in checkpoint -- is this a VideoViT checkpoint?"
        )
    fusion_in_dim = ckpt[fusion_key].shape[1]
    no_bank_dim   = VideoViT.NUM_TEMPORAL_HEADS * VideoViT.EMBED_DIM + 2
    bank_dim      = no_bank_dim + VideoViT.NUM_TEMPORAL_HEADS
    if fusion_in_dim == bank_dim:
        use_memory_bank = True
    elif fusion_in_dim == no_bank_dim:
        use_memory_bank = False
    else:
        raise ValueError(
            f"Unexpected fusion_classifier input dim {fusion_in_dim}. "
            f"Expected {no_bank_dim} (no bank) or {bank_dim} (with bank)."
        )
    print(f"  ✓ fusion_classifier input dim={fusion_in_dim} -> "
          f"use_memory_bank={use_memory_bank}")

    model = VideoViT(
        num_frames      = num_frames,
        use_memory_bank = use_memory_bank,
    ).to(device)

    # NOTE: if use_memory_bank=True, the bank must be built and attached in
    # main() before running inference (we do NOT attach a dummy bank here).
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("  Checkpoint loaded successfully.")
    return model, use_memory_bank


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Resolve CSV path
    cdfv3_csv = args.cdfv3_csv if args.cdfv3_csv else \
        str(Path(args.cdfv3_root) / "manifest_cdfv3_face_crops.csv")

    print(f"\n  Device      : {device}")
    print(f"  Model       : Frame-end VideoViT  (frame_model.ViT backbone, EMBED_DIM=1024, "
          f"layers tapped: {_TAPPED_LAYERS})")
    print(f"  Temporal heads : {VideoViT.NUM_TEMPORAL_HEADS}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  CDFv3 root  : {args.cdfv3_root}")
    print(f"  CDFv3 CSV   : {cdfv3_csv}")
    print(f"  Num frames  : {args.num_frames}  (per video, temporal inference)")
    print(f"  Batch size  : {args.batch_size}  (frame-level)")
    print(f"  Aggregation : {args.agg}  (legacy flag; all mean-based strategies evaluated)")
    print(f"  Top-k       : {args.topk}")
    print(f"  MAC head    : index {args.mac_head_idx}  "
          f"(layer {_TAPPED_LAYERS[args.mac_head_idx]})")
    print(f"  Precision   : {'FP32 (--fp32)' if args.fp32 else 'FP16 autocast'}")
    print(f"  Real bias   : {args.real_bias}"
          + ("  (disabled)" if args.real_bias == 0.0 else
             f"  → exponent = {1.0 + args.real_bias:.2f} applied to scores < 0.5"))
    if args.train_real_root:
        print(f"  Train real root : {args.train_real_root}")
        print(f"  kNN k           : {args.knn_k}")
    if args.show_misclassified:
        print(f"  Misclassified: will save grid → {args.misclassified_out}  "
              f"(max {args.max_display}, threshold={args.misclassified_threshold})")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    frame_dataset = CDFv3Dataset(cdfv3_csv, args.cdfv3_root)
    clip_dataset  = CDFv3ClipDataset(cdfv3_csv, args.cdfv3_root, num_frames=args.num_frames)

    _persistent = args.num_workers > 0
    _prefetch   = 4 if args.num_workers > 0 else None

    frame_loader = DataLoader(
        frame_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )

    clip_batch_size = max(1, args.batch_size // args.num_frames)
    clip_loader = DataLoader(
        clip_dataset,
        batch_size  = clip_batch_size,
        shuffle     = False, num_workers=args.num_workers,
        pin_memory  = True, collate_fn=clip_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"\n  Loading VideoViT (num_frames={args.num_frames}) …")
    model, use_memory_bank = load_model(args.checkpoint, args.num_frames, device)

    if use_memory_bank:
        if not args.train_real_root:
            raise ValueError(
                "This checkpoint was trained with --use_memory_bank.\n"
                "Please provide --train_real_root so the kNN bank can be rebuilt "
                "from real training data.\n"
                "Example:\n"
                "  --train_real_root /path/to/ffpp/preprocessed_out/real"
            )
        bank = build_memory_bank_for_inference(
            model           = model,
            real_train_root = Path(args.train_real_root),
            num_frames      = args.num_frames,
            k               = args.knn_k,
            batch_size      = args.bank_batch_size,
            num_workers     = args.num_workers,
            device          = device,
        )
        model.attach_memory_bank(bank)

    if not args.no_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── (A): frame-level inference ───────────────────────────────────────────
    print("\n  Running frame-level inference …")
    frame_labels, frame_probs, frame_logits, frame_vids, frame_paths = run_frame_inference(
        model, frame_loader, device,
        mac_head_idx=args.mac_head_idx,
        use_fp32=args.fp32,
    )
    print(f"  Total frames evaluated: {len(frame_labels)}")

    # ── Remap manifest label convention to metric convention ─────────────────
    # Manifest: label=1 → Real, label=0 → Fake
    # Metrics / model output: 0 → Real, 1 → Fake  (P(fake) = frame_probs[:,1])
    frame_labels = [1 - lbl for lbl in frame_labels]

    # ── Frame-level metrics ─────────────────────────────────────────────────
    auc_frame = compute_and_print_metrics(
        frame_labels, frame_probs, "(A) Frame-level  (CDFv3)"
    )

    # ── Misclassified image grid (frame-level) ───────────────────────────────
    if args.show_misclassified:
        display_misclassified(
            frame_labels, frame_probs, frame_vids, frame_paths,
            out_path=args.misclassified_out,
            threshold=args.misclassified_threshold,
            max_display=args.max_display,
        )

    # ── (B): video-mean aggregation ──────────────────────────────────────────
    (video_ids, video_labels,
     vid_mean_probs, vid_mean_logits, vid_topk_mean) = aggregate_to_video(
        frame_labels, frame_probs, frame_logits, frame_vids,
        agg=args.agg, topk=args.topk,
    )
    print(f"\n  Videos after frame aggregation: {len(video_ids)}")
    print(f"    Real  : {sum(l == 0 for l in video_labels)}")
    print(f"    Fake  : {sum(l == 1 for l in video_labels)}")

    # ── (C): video-temporal (clip) inference ─────────────────────────────────
    print("\n  Running clip-level (temporal) inference …")
    vid_ids_clip, vid_labels_clip, vid_frame_end_probs, vid_no_frame_probs = run_clip_inference(
        model, clip_loader, device, use_fp32=args.fp32,
    )
    print(f"  Total videos (temporal): {len(vid_ids_clip)}")

    # ── Apply real-score bias (video level only) ──────────────────────────
    bias_tag = ""
    if args.real_bias != 0.0:
        bias_tag = f"  [real_bias={args.real_bias}]"
        vid_mean_probs  = apply_real_bias(vid_mean_probs,  args.real_bias)
        vid_mean_logits = apply_real_bias(vid_mean_logits, args.real_bias)
        vid_topk_mean   = apply_real_bias(vid_topk_mean,   args.real_bias)
        vid_frame_end_probs = apply_real_bias(vid_frame_end_probs, args.real_bias)
        vid_no_frame_probs  = apply_real_bias(vid_no_frame_probs,  args.real_bias)
        print(f"\n  Real-score bias applied (real_bias={args.real_bias}, "
              f"exponent={1.0 + args.real_bias:.2f}).  "
              f"Scores < 0.5 suppressed; scores >= 0.5 unchanged.")

    # ── Video-level metrics ─────────────────────────────────────────────────
    auc_mean_probs  = compute_and_print_metrics(
        video_labels, vid_mean_probs,
        f"(B) Video-mean probs{bias_tag}  (CDFv3)"
    )
    auc_mean_logits = compute_and_print_metrics(
        video_labels, vid_mean_logits,
        f"(B) Video-mean logits→sigmoid{bias_tag}  (CDFv3)"
    )
    auc_topk_mean   = compute_and_print_metrics(
        video_labels, vid_topk_mean,
        f"(B) Video-top{args.topk}-mean probs{bias_tag}  (CDFv3)"
    )
    auc_frame_end = compute_and_print_metrics(
        vid_labels_clip, vid_frame_end_probs,
        f"(C) Video-temporal frame-end{bias_tag}  (CDFv3)"
    )
    auc_no_frame = compute_and_print_metrics(
        vid_labels_clip, vid_no_frame_probs,
        f"(D) Video-temporal no-frame{bias_tag}  (CDFv3)"
    )

    # ── Optional per-video CSV ───────────────────────────────────────────────
    if args.save_results:
        clip_frame_end = {
            vid: prob for vid, prob in zip(vid_ids_clip, vid_frame_end_probs)
        }
        clip_no_frame = {
            vid: prob for vid, prob in zip(vid_ids_clip, vid_no_frame_probs)
        }
        rows = []
        for i, vid in enumerate(video_ids):
            rows.append({
                "video_id":        vid,
                "label":           video_labels[i],
                "mean_prob":       vid_mean_probs[i],
                "mean_logit_prob": vid_mean_logits[i],
                f"top{args.topk}_mean_prob": vid_topk_mean[i],
                "frame_end_prob":  clip_frame_end.get(vid, float("nan")),
                "no_frame_prob":   clip_no_frame.get(vid, float("nan")),
            })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.save_results, index=False)
        print(f"\n  Per-video results written to: {args.save_results}")

    # ── Summary ─────────────────────────────────────────────────────────────
    sep = "═" * 72
    print(f"\n  {sep}")
    print(f"  FINAL SUMMARY  [VideoViT  num_frames={args.num_frames}]{bias_tag}")
    print(f"  {sep}")
    print(f"  Model         : frame_model.ViT backbone + frame-end classifier (EMBED_DIM=1024)")
    print(f"  MAC head used : index {args.mac_head_idx}  "
          f"(layer {_TAPPED_LAYERS[args.mac_head_idx]})")
    print(f"  {sep}")
    print(f"  (A) Frame-level AUC                        : {auc_frame:.4f}")
    print(f"  (B) Video-mean probs AUC                   : {auc_mean_probs:.4f}")
    print(f"  (B) Video-mean logits→sigmoid AUC          : {auc_mean_logits:.4f}")
    print(f"  (B) Video-top{args.topk}-mean probs AUC          : {auc_topk_mean:.4f}")
    print(f"  (C) Video-temporal frame-end AUC           : {auc_frame_end:.4f}  <- primary")
    print(f"  (D) Video-temporal no-frame AUC            : {auc_no_frame:.4f}")
    print(f"  {sep}")
    if args.show_misclassified:
        print(f"  Misclassified grid               : {args.misclassified_out}")
        print(f"  {sep}")
    print()


if __name__ == "__main__":
    main()
