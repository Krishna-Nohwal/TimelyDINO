"""
cdfv2_test_video.py  —  Video-level AUC evaluation on CDFv2 using a Stage 2
                         VideoViT checkpoint  (video_model_knn.VideoViT).

Updated for the new Stage 1 backbone (frame_42.ViT):
  - frame_42.ViT taps 5 layers [19,20,21,22,23]  (was 4 layers [20,21,22,23])
  - ViT.forward returns (logits_list, features_list, cls_list, fused_list)
    — a 4-tuple (was a 3-tuple)
  - cls_list entries are (B*T, EMBED_DIM) f_cls tensors, already squeezed
  - Deepest head is now index 4 (was index 3)
  - VideoViT.NUM_HEADS renamed to VideoViT.NUM_TEMPORAL_HEADS = 5 (was 4)
  - fusion_classifier input dim: 5122 (no bank) / 5127 (with bank)
    (was 4098 / 4102)

Three metrics are reported:
  (A) Frame-level         — per-frame prob from deepest SpatialHead (layer 23,
                            index 4), identical to Stage 1 / frame-only scripts.
  (B) Video-mean          — mean / top-k mean of valid-frame SpatialHead probs,
                            aggregated per video (no temporal transformer).
  (C) Video-temporal      — fused output of temporal transformers +
                            fusion_classifier (the Stage 2 trained head).

Usage
-----
python cdfv2_test_video.py \
    --checkpoint  checkpoints_s2/best.pth \
    --fake_root   /path/to/preprocessed_cdfv2/fake/cdfv2 \
    --real_root   /path/to/preprocessed_cdfv2/real \
    [--num_frames 32] [--batch_size 4] [--num_workers 4] \
    [--topk 10] [--no_compile] [--fp32] \
    [--real_bias 0.0] [--save_results results.csv]

Directory layout
----------------
  <fake_root>/<sample_dir>/image.png
  <real_root>/<sample_dir>/image.png

  Sample dir names encode video + frame via a trailing _fNNNN token:
      id10_id11_0001_f0011  ->  video id "id10_id11_0001"
      00011_f0052           ->  video id "00011"
"""

import re
import argparse
import math
import numpy as np
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
from video_model_knn42 import VideoViT, RealVideoMemoryBank


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a Stage 2 VideoViT checkpoint on CDFv2."
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to Stage 2 .pth checkpoint (full VideoViT state dict).")
    p.add_argument("--fake_root",
                   default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2",
                   help="Root dir containing per-frame fake sample dirs.")
    p.add_argument("--real_root",
                   default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real",
                   help="Root dir containing per-frame real sample dirs.")
    p.add_argument("--num_frames",   default=32,  type=int,
                   help="Frames sampled per video (must match Stage 2 training).")
    p.add_argument("--batch_size",   default=4,   type=int,
                   help="Number of videos per batch.")
    p.add_argument("--num_workers",  default=4,   type=int)
    p.add_argument("--topk",         default=10,  type=int,
                   help="k for top-k mean aggregation in (B) video-mean.")
    p.add_argument("--no_compile",   action="store_true",
                   help="Skip torch.compile.")
    p.add_argument("--fp32",         action="store_true",
                   help="Run in FP32 instead of FP16 autocast.")
    p.add_argument("--real_bias",    default=0.0, type=float,
                   help="Power-transform bias applied to video scores < 0.5 to "
                        "suppress FPR. 0.0 = disabled. AUC is unaffected.")
    p.add_argument("--save_results", default="",
                   help="If given, write a per-video CSV to this path.")
    p.add_argument("--train_real_root", default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real", type=str,
                   help="Root dir of real training frames (required when checkpoint "
                        "uses use_memory_bank=True, to rebuild the real-video kNN bank). "
                        "Same layout as --real_root: <train_real_root>/<sample_dir>/image.png")
    p.add_argument("--knn_k",           default=32,  type=int,
                   help="Number of nearest real neighbours for kNN bank (default: 32).")
    p.add_argument("--bank_batch_size", default=16,  type=int,
                   help="Batch size for building the kNN memory bank (default: 16).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 256   # must match training

# Deepest SpatialHead index in frame_42.ViT (5 tapped layers [19,20,21,22,23],
# 0-based -> deepest = index 4, corresponding to layer 23).
DEEPEST_HEAD_IDX = 4


# ---------------------------------------------------------------------------
# Video ID helper
# ---------------------------------------------------------------------------

def video_id_from_sample(sample_name: str) -> str:
    """
    Strip trailing frame-index suffix to get video ID.

    Handles both naming conventions seen across datasets:
      'id10_id11_0001_f0011'  -> 'id10_id11_0001'   (CDFv2: '_fNNNN')
      '000_frame_00'          -> '000'              (FF++ train: '_frame_NN')
    """
    return re.sub(r'_(?:frame_|f)\d+$', '', sample_name)


def collect_samples(root: Path, label: int):
    samples = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "image.png").exists():
            samples.append((d, label))
    return samples


# ---------------------------------------------------------------------------
# Frame dataset  (for stream A/B inference via frame_model directly)
# ---------------------------------------------------------------------------

class CDFv2FrameDataset(Dataset):
    """
    Loads individual frames.  __getitem__ returns (image, label, video_id)
    so the eval loop can group frames into videos without a second pass.
    """

    def __init__(self, fake_root: Path, real_root: Path):
        fake_samples = collect_samples(fake_root, label=1)
        real_samples = collect_samples(real_root, label=0)
        print(f"  CDFv2 frames  ->  Real: {len(real_samples)}  |  "
              f"Fake: {len(fake_samples)}  |  Total: {len(fake_samples) + len(real_samples)}")
        self.samples = fake_samples + real_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d, label = self.samples[idx]
        img      = load_and_resize(str(d / "image.png"), IMG_SIZE)
        img      = normalize(img)
        video_id = video_id_from_sample(d.name)
        return img, label, video_id


# ---------------------------------------------------------------------------
# Clip dataset  (for stream C — full VideoViT temporal inference)
# ---------------------------------------------------------------------------

def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    """Uniform stride; tiles if fewer frames than needed."""
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


class CDFv2ClipDataset(Dataset):
    """
    Groups per-frame directories into videos, samples num_frames per video.
    Used for the temporal transformer forward pass (stream C).
    """

    def __init__(self, fake_root: Path, real_root: Path, num_frames: int):
        self.num_frames = num_frames

        vid2paths: dict = defaultdict(list)
        vid2label: dict = {}

        for root, label in [(fake_root, 1), (real_root, 0)]:
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "image.png").exists():
                    vid = video_id_from_sample(d.name)
                    vid2paths[vid].append(str(d / "image.png"))
                    vid2label[vid] = label

        self.videos = []
        for vid, paths in sorted(vid2paths.items()):
            self.videos.append((vid, sorted(paths), vid2label[vid]))

        real_n = sum(1 for _, _, l in self.videos if l == 0)
        fake_n = sum(1 for _, _, l in self.videos if l == 1)
        print(f"  CDFv2 clips   ->  Real: {real_n}  |  Fake: {fake_n}  |  "
              f"Total: {len(self.videos)}")

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_and_print_metrics(labels, probs, level: str) -> float:
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

    cm             = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sep = "\u2500" * 72
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


def apply_real_bias(probs: list, real_bias: float) -> list:
    """Monotone power-transform suppressing scores < 0.5. AUC-preserving."""
    if real_bias == 0.0:
        return probs
    exp = 1.0 + real_bias
    return [p ** exp if p < 0.5 else p for p in probs]


# ---------------------------------------------------------------------------
# Inference  --  (A) frame-level and (B) video-mean via frame loader
# ---------------------------------------------------------------------------

def run_frame_inference(model: VideoViT, loader: DataLoader, device: torch.device,
                        use_fp32: bool = False, topk: int = 10):
    """
    Single-frame forward pass through the frozen frame_model only.
    Returns per-frame lists and video-level aggregations for streams (A) and (B).
    """
    frame_labels: list = []
    frame_probs:  list = []
    frame_logits: list = []
    frame_vids:   list = []

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
        for imgs, labels, video_ids in tqdm(loader, desc="Frame inference", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)

            # frame_42.ViT returns (logits_list, features_list, cls_list, fused_list)
            logits_list, _, _, _ = frame_model(imgs)
            raw_logits = logits_list[DEEPEST_HEAD_IDX].float()       # (B, 2)
            probs      = torch.softmax(raw_logits, dim=1)[:, 1].cpu().numpy()
            fake_logit = raw_logits[:, 1].cpu().numpy()

            frame_probs.extend(probs.tolist())
            frame_logits.extend(fake_logit.tolist())
            frame_labels.extend(labels.numpy().tolist())
            frame_vids.extend(list(video_ids))

    # Aggregate per video.
    vid2labels:  dict = defaultdict(list)
    vid2probs:   dict = defaultdict(list)
    vid2logits:  dict = defaultdict(list)

    for lbl, prob, logit, vid in zip(frame_labels, frame_probs, frame_logits, frame_vids):
        vid2labels[vid].append(lbl)
        vid2probs[vid].append(prob)
        vid2logits[vid].append(logit)

    video_ids_sorted = sorted(vid2labels.keys())
    video_labels     = []
    vid_mean_probs   = []
    vid_mean_logits  = []
    vid_topk_mean    = []

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    for vid in video_ids_sorted:
        lbls   = vid2labels[vid]
        unique = set(lbls)
        if len(unique) > 1:
            print(f"  [WARNING] Video '{vid}' has mixed labels {unique}; using majority.")
        video_labels.append(int(round(np.mean(lbls))))

        frame_p = np.array(vid2probs[vid])
        frame_l = np.array(vid2logits[vid])

        vid_mean_probs.append(float(frame_p.mean()))
        vid_mean_logits.append(float(_sigmoid(frame_l.mean())))

        k = min(topk, len(frame_p))
        topk_probs = np.partition(frame_p, -k)[-k:]
        vid_topk_mean.append(float(topk_probs.mean()))

    return (
        frame_labels, frame_probs,
        video_ids_sorted, video_labels,
        vid_mean_probs, vid_mean_logits, vid_topk_mean,
    )


# ---------------------------------------------------------------------------
# Inference  --  (C) video-temporal via clip loader
# ---------------------------------------------------------------------------

def run_clip_inference(model: VideoViT, loader: DataLoader, device: torch.device,
                       use_fp32: bool = False):
    """
    Full VideoViT forward pass through temporal transformers + fusion_classifier.
    Returns (video_ids, video_labels, video_temp_probs).
    """
    video_ids_out:    list = []
    video_labels_out: list = []
    video_temp_probs: list = []

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

            video_logits, _, _, _ = model(frames, lengths)
            probs = torch.softmax(video_logits.float(), dim=1)[:, 1].cpu().numpy()

            video_temp_probs.extend(probs.tolist())
            video_labels_out.extend(labels.numpy().tolist())
            video_ids_out.extend(vids)

    return video_ids_out, video_labels_out, video_temp_probs



# ---------------------------------------------------------------------------
# Training real-video dataset  (for kNN bank construction at inference time)
# ---------------------------------------------------------------------------

class TrainRealClipDataset(Dataset):
    """
    Loads only real training video clips for kNN memory bank construction.

    Uses the same directory-scanning approach as CDFv2ClipDataset so frame
    selection is consistent. No augmentation — we want clean embeddings.

    Args
    ----
    real_train_root : Path — root dir of real training frames.
                      Same layout as --real_root: <root>/<sample_dir>/image.png
                      Sample dirs encode video + frame via trailing _fNNNN.
    num_frames      : int  — frames to sample per video (must match training).
    """

    def __init__(self, real_train_root: Path, num_frames: int):
        self.num_frames = num_frames

        vid2paths: dict = defaultdict(list)

        for d in sorted(real_train_root.iterdir()):
            if d.is_dir() and (d / "image.png").exists():
                vid = video_id_from_sample(d.name)
                vid2paths[vid].append(str(d / "image.png"))

        self.videos = [(vid, sorted(paths)) for vid, paths in sorted(vid2paths.items())]
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

    Mirrors build_memory_bank() from train_stage2.py exactly — calls
    frame_model and temporal_transformers directly (not the full forward pass)
    to avoid the chicken-and-egg problem of querying a bank that doesn't exist.

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
        num_heads = VideoViT.NUM_TEMPORAL_HEADS,   # 5 (was VideoViT.NUM_HEADS = 4)
        k         = k,
    )

    # Unwrap torch.compile if active.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    was_training = model.training
    model.eval()

    with torch.inference_mode(), \
         torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, _, lengths in tqdm(loader, desc="  Building bank", leave=False):
            B, T, C, H, W = frames.shape
            frames  = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            # Call frame_model and temporal transformers directly —
            # do NOT call model() which requires the bank to already exist.
            flat_frames = frames.reshape(B * T, C, H, W)

            # frame_42.ViT returns (logits_list, features_list, cls_list, fused_list)
            # cls_list[i] : (B*T, EMBED_DIM)  — f_cls, already squeezed
            _, _, cls_list, _ = raw_model.frame_model(flat_frames)

            time_idx         = torch.arange(T, device=device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.unsqueeze(1)

            video_feats_list = []
            for temporal_tfm, cls_tokens in zip(raw_model.temporal_transformers, cls_list):
                frame_cls = cls_tokens.reshape(B, T, raw_model.EMBED_DIM)
                video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

            bank.add([v.float() for v in video_feats_list])

    bank.build()

    if was_training:
        model.train()

    print(f"  Memory bank ready: {len(bank)} real video embeddings, k={k}")
    return bank


# ---------------------------------------------------------------------------
# Model loading helper  (auto-detects use_memory_bank from checkpoint shape)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, num_frames: int, device: torch.device) -> VideoViT:
    """
    Load a VideoViT checkpoint, auto-detecting use_memory_bank from the
    fusion_classifier weight shape so the model is always instantiated with
    the same architecture that was saved.

    When use_memory_bank=True is detected, a dummy zero bank is attached so
    that VideoViT.forward() can call memory_bank.query() without needing the
    full training set on disk.  The kNN similarity scores are fixed at 0.0,
    which is a constant offset absorbed by the trained fusion_classifier weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}

    # Shape check on frame_model SpatialHead head[0] (frame_42.SpatialHead.head)
    # head : nn.Sequential(Linear(2*embed_dim, embed_dim//2), ReLU, Dropout)
    # so head.0.weight has shape (embed_dim//2, 2*embed_dim) = (512, 2048)
    fc1_key = "frame_model.spatial_heads.0.head.0.weight"
    if fc1_key in ckpt:
        expected = (512, 2048)
        actual   = tuple(ckpt[fc1_key].shape)
        if actual != expected:
            raise ValueError(
                f"Checkpoint appears to be from a different model size.\n"
                f"  Expected {fc1_key} shape : {expected}  (ViT-Large)\n"
                f"  Got                      : {actual}"
            )
        print(f"  \u2713 frame_model SpatialHead shape check passed {actual}")
    else:
        print(f"  [WARNING] Could not find '{fc1_key}' in checkpoint for shape check.")

    # Auto-detect use_memory_bank from fusion_classifier input dim:
    #   5122 = NUM_TEMPORAL_HEADS*EMBED_DIM + 2                  -> use_memory_bank=False
    #   5127 = NUM_TEMPORAL_HEADS*EMBED_DIM + 2 + NUM_TEMPORAL_HEADS -> use_memory_bank=True
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
    print(f"  \u2713 fusion_classifier input dim={fusion_in_dim} -> "
          f"use_memory_bank={use_memory_bank}")

    model = VideoViT(
        num_frames      = num_frames,
        use_memory_bank = use_memory_bank,
    ).to(device)

    # NOTE: if use_memory_bank=True, the bank must be built and attached in
    # main() before running inference. load_model() returns use_memory_bank
    # so main() knows whether to call build_memory_bank_for_inference().
    # We do NOT attach a dummy bank here — that would corrupt predictions.

    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("  Checkpoint loaded successfully.")
    return model, use_memory_bank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    fake_root = Path(args.fake_root)
    real_root = Path(args.real_root)

    sep = "\u2550" * 72
    print(f"\n  {sep}")
    print(f"  CDFv2 EVALUATION  --  VideoViT (Stage 2)")
    print(f"  {sep}")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Fake root   : {fake_root}")
    print(f"  Real root   : {real_root}")
    print(f"  Num frames  : {args.num_frames}  (per video, temporal inference)")
    print(f"  Batch size  : {args.batch_size}  videos")
    print(f"  Top-k       : {args.topk}")
    print(f"  Precision   : {'FP32 (--fp32)' if args.fp32 else 'FP16 autocast'}")
    print(f"  Temporal heads : {VideoViT.NUM_TEMPORAL_HEADS}  (frame_42.ViT, layers [19-23])")
    print(f"  Real bias   : {args.real_bias}"
          + ("  (disabled)" if args.real_bias == 0.0 else
             f"  -> exponent {1.0 + args.real_bias:.2f} applied to scores < 0.5"))
    if args.train_real_root:
        print(f"  Train real root : {args.train_real_root}")
        print(f"  kNN k           : {args.knn_k}")

    # -- Datasets & loaders --------------------------------------------------
    frame_dataset = CDFv2FrameDataset(fake_root, real_root)
    clip_dataset  = CDFv2ClipDataset(fake_root, real_root, num_frames=args.num_frames)

    _persistent = args.num_workers > 0
    _prefetch   = 4 if args.num_workers > 0 else None

    frame_loader = DataLoader(
        frame_dataset,
        batch_size  = args.batch_size * args.num_frames,
        shuffle     = False, num_workers=args.num_workers,
        pin_memory  = True, persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    clip_loader = DataLoader(
        clip_dataset,
        batch_size  = args.batch_size,
        shuffle     = False, num_workers=args.num_workers,
        pin_memory  = True, collate_fn=clip_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    # -- Model ---------------------------------------------------------------
    print(f"\n  Loading VideoViT (num_frames={args.num_frames}) ...")
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
        print("  Compiling model with torch.compile ...")
        model = torch.compile(model)

    # -- (A) & (B): frame inference ------------------------------------------
    print("\n  Running frame-level inference ...")
    (frame_labels, frame_probs,
     vid_ids_frame, vid_labels_frame,
     vid_mean_probs, vid_mean_logits, vid_topk_mean) = run_frame_inference(
        model, frame_loader, device,
        use_fp32=args.fp32, topk=args.topk,
    )
    print(f"  Total frames evaluated: {len(frame_labels)}")
    print(f"  Total videos (frame aggregation): {len(vid_ids_frame)}")

    # -- (C): clip / temporal inference --------------------------------------
    print("\n  Running clip-level (temporal) inference ...")
    vid_ids_clip, vid_labels_clip, vid_temp_probs = run_clip_inference(
        model, clip_loader, device, use_fp32=args.fp32,
    )
    print(f"  Total videos (temporal): {len(vid_ids_clip)}")

    # -- Apply real-score bias -----------------------------------------------
    bias_tag = ""
    if args.real_bias != 0.0:
        bias_tag        = f"  [real_bias={args.real_bias}]"
        vid_mean_probs  = apply_real_bias(vid_mean_probs,  args.real_bias)
        vid_mean_logits = apply_real_bias(vid_mean_logits, args.real_bias)
        vid_topk_mean   = apply_real_bias(vid_topk_mean,   args.real_bias)
        vid_temp_probs  = apply_real_bias(vid_temp_probs,  args.real_bias)
        print(f"\n  Real-score bias applied (exponent={1.0 + args.real_bias:.2f}).")

    # -- Print metrics -------------------------------------------------------
    auc_frame = compute_and_print_metrics(
        frame_labels, frame_probs,
        f"(A) Frame-level{bias_tag}  (CDFv2)"
    )
    auc_mean_probs = compute_and_print_metrics(
        vid_labels_frame, vid_mean_probs,
        f"(B) Video-mean probs{bias_tag}  (CDFv2)"
    )
    auc_mean_logits = compute_and_print_metrics(
        vid_labels_frame, vid_mean_logits,
        f"(B) Video-mean logits->sigmoid{bias_tag}  (CDFv2)"
    )
    auc_topk = compute_and_print_metrics(
        vid_labels_frame, vid_topk_mean,
        f"(B) Video-top{args.topk}-mean probs{bias_tag}  (CDFv2)"
    )
    auc_temporal = compute_and_print_metrics(
        vid_labels_clip, vid_temp_probs,
        f"(C) Video-temporal (fusion){bias_tag}  (CDFv2)"
    )

    # -- Optional per-video CSV ----------------------------------------------
    if args.save_results:
        import pandas as pd
        frame_side = {
            vid: {"label": lbl, "mean_prob": mp, "mean_logit_prob": ml,
                  f"top{args.topk}_mean_prob": tk}
            for vid, lbl, mp, ml, tk in zip(
                vid_ids_frame, vid_labels_frame,
                vid_mean_probs, vid_mean_logits, vid_topk_mean
            )
        }
        clip_side = {vid: prob for vid, prob in zip(vid_ids_clip, vid_temp_probs)}
        rows = []
        for vid in sorted(frame_side.keys()):
            row = {"video_id": vid}
            row.update(frame_side[vid])
            row["temporal_prob"] = clip_side.get(vid, float("nan"))
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.save_results, index=False)
        print(f"\n  Per-video results written to: {args.save_results}")

    # -- Summary -------------------------------------------------------------
    print(f"\n  {sep}")
    print(f"  FINAL SUMMARY  [VideoViT  num_frames={args.num_frames}]{bias_tag}")
    print(f"  {sep}")
    print(f"  (A) Frame-level AUC                        : {auc_frame:.4f}")
    print(f"  (B) Video-mean probs AUC                   : {auc_mean_probs:.4f}")
    print(f"  (B) Video-mean logits->sigmoid AUC         : {auc_mean_logits:.4f}")
    print(f"  (B) Video-top{args.topk}-mean probs AUC          : {auc_topk:.4f}")
    print(f"  (C) Video-temporal (fusion) AUC            : {auc_temporal:.4f}  <- primary")
    print(f"  {sep}")
    print()


if __name__ == "__main__":
    main()