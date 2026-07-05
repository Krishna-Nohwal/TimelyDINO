"""
train_stage2_frame_end_no_memory.py

Stage 2 video-level training without any retrieval/prototype branch. The frozen
Stage 1 frame model provides per-frame logits and four tapped CLS streams. Four
temporal transformers process those streams, and the final classifier receives
the concatenated temporal representation plus the averaged deepest frame logits.

Trainable modules:
  - temporal_transformers
  - fusion_classifier

Frozen module:
  - frame_model

Outputs:
  checkpoints_s2_frame_end_no_memory/latest.pth
  checkpoints_s2_frame_end_no_memory/best.pth

Example:
python train_stage2_frame_end_no_memory.py \
    --load_from checkpoints_vit_4layers/best.pth \
    --manifest /path/to/manifest_ff_onct.csv \
    --root_dir /path/to/preprocessed_ffpp \
    --cdf_root /path/to/cdfv1_onct_out \
    --cdf_csv /path/to/cdfv1_onct_out/manifest_cdfv1_onct.csv \
    --num_frames 32 \
    --batch_size 10
"""

import os, math, argparse, csv
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score,
)
from augmentations import augment_batch, load_and_resize, normalize
from frame_model import ViT
from video_model import TemporalTransformer, temporal_augment


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Stage 2: train temporal transformers with frame logits appended at the final classifier"
)
parser.add_argument("--epochs",           default=30,   type=int)
parser.add_argument("--batch_size",       default=10,   type=int,
                    help="Videos per batch. Must be even for balanced sampler.")
parser.add_argument("--num_frames",       default=32,   type=int,
                    help="Frames sampled per video (uniform stride).")
parser.add_argument("--num_workers",      default=20,   type=int)
parser.add_argument("--save_root",        default="checkpoints_s2_frame_end_no_memory", type=str)
parser.add_argument("--load_from",        default="",   type=str,
                    help="Path to Stage 1 best.pth (required).")
parser.add_argument("--manifest",         default="E:/Work/sampled_30k/manifest_onct.csv", type=str)
parser.add_argument("--root_dir",         default="E:/Work/sampled_30k/", type=str)
parser.add_argument("--cdf_root",         default="E:/Work/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv",          default="E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
parser.add_argument("--val_ratio",        default=0.05, type=float,
                    help="Fraction of videos held out for validation.")
parser.add_argument("--lr",               default=1e-3, type=float,
                    help="Base LR for temporal transformers + fusion_classifier.")
parser.add_argument("--warmup_steps",     default=64,   type=int)
parser.add_argument("--supcon_weight",    default=1/16, type=float)
parser.add_argument("--no_compile",       action="store_true",
                    help="Disable torch.compile (useful for debugging).")
parser.add_argument("--disable_tqdm",     action="store_true",
                    help="Disable tqdm progress bars for cleaner redirected/grid-search logs.")
parser.add_argument("--metrics_csv",      default="", type=str,
                    help="Optional CSV path for grid-search metric summaries.")
parser.add_argument("--run_name",         default="", type=str,
                    help="Run name stored in --metrics_csv rows.")
args = parser.parse_args()

if args.disable_tqdm:
    tqdm = partial(tqdm, disable=True)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

IMG_SIZE     = 256
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Video model with frame-logit shortcut restored
# ---------------------------------------------------------------------------

class VideoViT(nn.Module):
    """
    Stage 2 video model with the old frame-logit shortcut restored.

    The final classifier receives:
      - temporal_vec: 4 temporal heads x 1024 = 4096 dims
      - frame_mean_logits: deepest frame head logits averaged over valid frames = 2 dims

    During eval we also report an ablation where the final two frame-logit
    dimensions are zeroed before the same classifier.
    """

    EMBED_DIM          = ViT.EMBED_DIM
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS

    def __init__(
        self,
        num_frames:       int   = 32,
        temporal_layers:  int   = 2,
        temporal_heads:   int   = 8,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames      = num_frames

        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim  = self.EMBED_DIM,
                num_frames = num_frames,
                num_layers = temporal_layers,
                num_heads  = temporal_heads,
                dropout    = temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])

        self.fusion_classifier = nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM + 2, 2
        )

    @property
    def vit(self):
        return self.frame_model.vit

    @staticmethod
    def _mean_valid_frame_logits(
        frame_logits_list: list,
        B: int,
        T: int,
        key_padding_mask: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        frame_logits = frame_logits_list[-1].float().reshape(B, T, 2)

        if key_padding_mask is None:
            return frame_logits.mean(dim=1).to(dtype=dtype)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        counts = valid.sum(dim=1).clamp(min=1)
        return ((frame_logits * valid).sum(dim=1) / counts).to(dtype=dtype)

    def forward(
        self,
        video: Tensor,
        lengths: Tensor | None = None,
    ):
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

        video_feats_list = []
        for temporal_tfm, frame_cls in zip(self.temporal_transformers, cls_sequences):
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)
        frame_mean_logits = self._mean_valid_frame_logits(
            frame_logits_list, B, T, key_padding_mask, temporal_vec.dtype
        )

        fused_with_frame = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        fused_no_frame = torch.cat([temporal_vec, torch.zeros_like(frame_mean_logits)], dim=1)

        video_logits_with_frame = self.fusion_classifier(fused_with_frame)
        video_logits_no_frame = self.fusion_classifier(fused_no_frame)

        return (
            video_logits_with_frame,
            video_logits_no_frame,
            frame_logits_list,
            frame_feats_list,
            video_feats_list,
        )

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        ckpt  = torch.load(image_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))

        frame_state   = {}
        has_fm_prefix = any(k.startswith("frame_model.") for k in state)

        for key, value in state.items():
            if key.startswith("frame_model."):
                frame_state[key[len("frame_model."):]] = value
            elif key.startswith(("temporal_transformers.", "fusion_classifier.")):
                pass
            else:
                if not has_fm_prefix:
                    frame_state[key] = value

        missing, unexpected = self.frame_model.load_state_dict(frame_state, strict=strict)
        print(f"Loaded image weights - missing: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_probs, split_name: str, epoch: int) -> float:
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    ap  = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer     = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2

    cm             = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"  [{split_name}] Epoch {epoch+1:02d} | "
        f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc*100:.2f}%  F1={f1:.4f}  "
        f"EER={eer*100:.2f}%  TPR={tpr*100:.2f}%  FPR={fpr*100:.2f}%  "
        f"TNR={tnr*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return auc


def write_grid_metric_rows(epoch: int, split: str, metrics: dict):
    if not args.metrics_csv:
        return

    csv_path = Path(args.metrics_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = [
        "run_name", "epoch", "split", "mode", "auc",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for mode, auc in metrics.items():
            writer.writerow({
                "run_name": args.run_name,
                "epoch": epoch + 1,
                "split": split,
                "mode": mode,
                "auc": auc,
            })


# ---------------------------------------------------------------------------
# Data splits  (video-level — must match Stage 1 to prevent leakage)
# ---------------------------------------------------------------------------

def _extract_video_id(sample_dir: str) -> str:
    """
    Derive a video-level ID from a per-frame sample_dir.

    Handles two manifest formats:
      Training : 'fake/FaceSwap/922_898_frame_31'  -> 'fake/FaceSwap/922_898'
      CDF      : 'real/00011_f0052'                -> 'real/00011'
               : 'fake/cdf/id1_id6_0007_f0072'    -> 'fake/cdf/id1_id6_0007'
    """
    import re
    parts    = Path(sample_dir).parts
    basename = parts[-1]
    prefix   = "/".join(parts[:-1])

    idx = basename.rfind("_frame_")
    if idx != -1:
        clip_id = basename[:idx]
    else:
        m = re.search(r"_f\d+$", basename)
        clip_id = basename[:m.start()] if m else basename

    return f"{prefix}/{clip_id}" if prefix else clip_id


def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    """
    Video-level split using the same seed as Stage 1 so the train/val
    boundary is identical regardless of which script ran first.
    """
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    df["video_id"] = df["sample_dir"].apply(_extract_video_id)

    real_vids = df[df["label"] == 0]["video_id"].unique()
    fake_vids = df[df["label"] == 1]["video_id"].unique()

    rng = np.random.default_rng(42)
    real_vids = rng.permutation(real_vids)
    fake_vids = rng.permutation(fake_vids)

    print(f"Full dataset -> Real videos: {len(real_vids)} | Fake videos: {len(fake_vids)}")

    real_val_n = max(1, int(len(real_vids) * val_ratio))
    fake_val_n = max(1, int(len(fake_vids) * val_ratio))

    val_ids  = set(real_vids[:real_val_n]) | set(fake_vids[:fake_val_n])
    train_df = df[~df["video_id"].isin(val_ids)].reset_index(drop=True)
    val_df   = df[ df["video_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train -> {len(train_df)} frames "
          f"(real vids: {len(real_vids) - real_val_n}  fake vids: {len(fake_vids) - fake_val_n})")
    print(f"Val   -> {len(val_df)} frames "
          f"(real vids: {real_val_n}  fake vids: {fake_val_n})")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Video datasets
# ---------------------------------------------------------------------------

def _load_video_frames(frame_paths: list, img_size: int) -> torch.Tensor:
    frames = []
    for p in frame_paths:
        try:
            img = load_and_resize(p, img_size)
            img = normalize(img)
        except Exception:
            img = torch.zeros(3, img_size, img_size)
        frames.append(img)
    return torch.stack(frames, dim=0)   # (T, 3, H, W)


def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def video_collate_fn(batch):
    """
    Pads variable-length clips to the longest clip in the batch.

    Returns
    -------
    frames  : (B, T_max, C, H, W)  float32
    labels  : (B,)                  int64
    lengths : (B,)                  int64
    """
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        if pad_t > 0:
            f = F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded.append(f)

    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
    )


class ManifestVideoDataset(Dataset):
    """
    Stage 2 train/val dataset.  label: 0 = Real, 1 = Fake.

    Groups per-frame CSV rows by video_id, samples num_frames frames with
    uniform stride, and optionally applies augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        num_frames: int = 32,
        augment: bool = True,
    ):
        self.num_frames = num_frames
        self.augment    = augment
        self.videos: list = []

        root = Path(root_dir)
        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
            paths = []
            for rel in group["sample_dir"].str.replace("\\", "/", regex=False):
                img_path = root / rel / "image.png"
                if img_path.is_file():
                    paths.append(str(img_path))
            paths = sorted(paths)
            if not paths:
                continue
            self.videos.append((paths, label))

        real_n = sum(1 for _, l in self.videos if l == 0)
        fake_n = sum(1 for _, l in self.videos if l == 1)
        print(f"  [ManifestVideoDataset] {len(self.videos)} videos "
              f"({real_n} real, {fake_n} fake)")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        indices      = _sample_frame_indices(len(paths), self.num_frames)
        sampled      = [paths[i] for i in indices]
        frames       = _load_video_frames(sampled, IMG_SIZE)
        if self.augment:
            frames = augment_batch(frames)
        return frames, label


class CDFv1VideoDataset(Dataset):
    """CDFv1 test dataset (video-level). No augmentation at test time."""

    def __init__(self, csv_path: str, data_root: str, num_frames: int = 32):
        self.num_frames = num_frames

        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"]    = df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(_extract_video_id)

        print(f"CDFv1 frames -> Real: {(df['label']==0).sum()} | "
              f"Fake: {(df['label']==1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        self.videos: list = []

        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
            paths = []
            for rel in group["sample_dir"].str.replace("\\", "/", regex=False):
                img_path = root / rel / "image.png"
                if img_path.is_file():
                    paths.append(str(img_path))
            paths = sorted(paths)
            if not paths:
                continue
            self.videos.append((paths, label))

        total_vids   = df["video_id"].nunique()
        skipped_vids = total_vids - len(self.videos)
        if skipped_vids:
            print(f"  [CDFv1] Skipped {skipped_vids} videos with no frames on disk")
        print(f"  [CDFv1] {len(self.videos)} videos loaded.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        indices      = _sample_frame_indices(len(paths), self.num_frames)
        sampled      = [paths[i] for i in indices]
        frames       = _load_video_frames(sampled, IMG_SIZE)
        return frames, label


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

class BalancedRealFakeBatchSampler(Sampler):
    """
    Yields class-balanced batches.  Each batch has exactly
    batch_size // 2 real and batch_size // 2 fake videos.
    batch_size must be even.
    """

    def __init__(self, dataset: ManifestVideoDataset, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("BalancedRealFakeBatchSampler requires an even batch_size.")
        self.per_class    = batch_size // 2
        self.real_indices = [i for i, (_, l) in enumerate(dataset.videos) if l == 0]
        self.fake_indices = [i for i, (_, l) in enumerate(dataset.videos) if l == 1]
        if not self.real_indices or not self.fake_indices:
            raise ValueError("Balanced sampler needs at least one real and one fake video.")
        self.num_batches = math.ceil(
            max(len(self.real_indices), len(self.fake_indices)) / self.per_class
        )

    @staticmethod
    def _oversample(indices: list, n: int) -> list:
        rng = np.random.default_rng()
        out, remaining = [], n
        while remaining > 0:
            perm = rng.permutation(indices).tolist()
            out.extend(perm[:remaining])
            remaining -= len(perm[:remaining])
        return out

    def __iter__(self):
        n         = self.num_batches * self.per_class
        real_pool = self._oversample(self.real_indices, n)
        fake_pool = self._oversample(self.fake_indices, n)
        rng       = np.random.default_rng()

        for i in range(self.num_batches):
            s     = i * self.per_class
            e     = s + self.per_class
            batch = real_pool[s:e] + fake_pool[s:e]
            yield rng.permutation(batch).tolist()

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

_bce_loss    = nn.CrossEntropyLoss()
_supcon_loss = SupConLoss()


def stage2_loss(
    video_logits: torch.Tensor,
    video_feats_list: list,
    labels: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    BCE on video logits + SupCon on each temporal head's embeddings.

    frame_model is frozen so frame terms are omitted for efficiency.

    video_logits     : (B, 2)
    video_feats_list : list of NUM_TEMPORAL_HEADS × (B, EMBED_DIM)
    labels           : (B,)
    """
    l_bce    = _bce_loss(video_logits, labels)
    l_supcon = lam * sum(_supcon_loss(feats, labels) for feats in video_feats_list)
    return l_bce + l_supcon


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _video_probs_from_forward(
    video_logits_with_frame: torch.Tensor,
    video_logits_no_frame: torch.Tensor,
    frame_logits_list: list,
    labels: torch.Tensor,
    lengths: torch.Tensor,
):
    """
    Derive three sets of predictions from one forward pass.

    (A) Frame-level   — per-frame fake prob from deepest SpatialHead (index 3)
    (B) Video-mean    — per-video mean of valid frame probs from (A)
    (C) Video-temporal — per-video prob from temporal transformer path

    Returns
    -------
    frame_labels     : (N_valid_frames,)
    frame_probs      : (N_valid_frames,)
    video_labels     : (B,)
    video_mean_probs : (B,)
    video_temp_probs : (B,)
    """
    B = labels.size(0)
    T = frame_logits_list[0].size(0) // B

    # Padding mask
    time_idx   = torch.arange(T, device=labels.device).unsqueeze(0)   # (1, T)
    valid_2d   = time_idx < lengths.to(labels.device).unsqueeze(1)    # (B, T)
    valid_flat = valid_2d.reshape(-1)                                  # (B*T,)

    # (A) Deepest SpatialHead = index 3 (4 heads, 0-based)
    frame_probs_all  = torch.softmax(frame_logits_list[3].float(), dim=1)[:, 1]  # (B*T,)
    frame_labels_all = labels.repeat_interleave(T)                                # (B*T,)
    frame_probs      = frame_probs_all[valid_flat]
    frame_labels     = frame_labels_all[valid_flat]

    # (B) Video-mean
    frame_probs_2d   = frame_probs_all.reshape(B, T)
    valid_counts     = lengths.to(labels.device).float()
    masked_sum       = (frame_probs_2d * valid_2d.float()).sum(dim=1)
    video_mean_probs = masked_sum / valid_counts.clamp(min=1)

    # (C)/(D) Final video classifier, with and without the frame-logit shortcut.
    video_with_frame_probs = torch.softmax(video_logits_with_frame.float(), dim=1)[:, 1]
    video_no_frame_probs   = torch.softmax(video_logits_no_frame.float(), dim=1)[:, 1]

    return (
        frame_labels,
        frame_probs,
        labels,
        video_mean_probs,
        video_with_frame_probs,
        video_no_frame_probs,
    )


def run_eval(model: nn.Module, loader: DataLoader, desc: str):
    """
    Evaluate VideoViT on one DataLoader.

    Returns
    -------
    (frame_labels, frame_probs, video_labels, video_mean_probs, video_temp_probs)
    """
    frame_labels_all, frame_probs_all = [], []
    video_labels_all = []
    video_mean_probs_all, video_with_frame_probs_all, video_no_frame_probs_all = [], [], []

    model.eval()
    with torch.inference_mode(), \
         torch.cuda.amp.autocast(dtype=torch.float16):
        for frames, labels, lengths in tqdm(loader, desc=desc, leave=False):
            frames  = frames.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            (
                video_logits_with_frame,
                video_logits_no_frame,
                frame_logits_list,
                _,
                _,
            ) = model(frames, lengths)

            fl, fp, vl, vmp, vwfp, vnfp = _video_probs_from_forward(
                video_logits_with_frame,
                video_logits_no_frame,
                frame_logits_list,
                labels,
                lengths,
            )
            frame_labels_all.extend(fl.cpu().numpy().tolist())
            frame_probs_all.extend(fp.cpu().numpy().tolist())
            video_labels_all.extend(vl.cpu().numpy().tolist())
            video_mean_probs_all.extend(vmp.cpu().numpy().tolist())
            video_with_frame_probs_all.extend(vwfp.cpu().numpy().tolist())
            video_no_frame_probs_all.extend(vnfp.cpu().numpy().tolist())

    return (
        frame_labels_all, frame_probs_all,
        video_labels_all, video_mean_probs_all,
        video_with_frame_probs_all, video_no_frame_probs_all,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  STAGE 2 — Temporal transformer training (frame_model FROZEN)")
    print(f"  Backbone: frame_model_4layers.ViT  |  Temporal heads: {VideoViT.NUM_TEMPORAL_HEADS}")
    print(f"{SEP}\n")

    if not args.load_from:
        raise ValueError(
            "--load_from is required for Stage 2. "
            "Point it to the Stage 1 best.pth checkpoint."
        )

    NUM_FRAMES = args.num_frames

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, val_df = prepare_splits(
        args.manifest, args.root_dir, val_ratio=args.val_ratio
    )

    train_dataset = ManifestVideoDataset(
        train_df, args.root_dir, num_frames=NUM_FRAMES, augment=True
    )
    val_dataset = ManifestVideoDataset(
        val_df, args.root_dir, num_frames=NUM_FRAMES, augment=False
    )
    cdf_dataset = CDFv1VideoDataset(
        args.cdf_csv, args.cdf_root, num_frames=NUM_FRAMES
    )

    train_batch_sampler = BalancedRealFakeBatchSampler(train_dataset, args.batch_size)
    print(
        f"Train balanced batches -> {len(train_batch_sampler)} batches/epoch "
        f"({args.batch_size // 2} real + {args.batch_size // 2} fake videos per batch)"
    )

    _persistent = _num_workers > 0
    _prefetch   = 4 if _num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    cdf_loader = DataLoader(
        cdf_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    os.makedirs(args.save_root, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = VideoViT(num_frames=NUM_FRAMES).to(device)

    print(f"Loading Stage 1 weights from: {args.load_from}")
    missing, unexpected = model.load_image_weights(args.load_from, strict=False)
    if missing:
        print(f"  Missing keys in frame_model: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    # ── Freeze frame_model completely ────────────────────────────────────────
    model.frame_model.requires_grad_(False)

    trainable   = [p for p in model.parameters() if p.requires_grad]
    total_n     = sum(p.numel() for p in model.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    print(
        f"\n  frame_model:              FROZEN\n"
        f"  temporal_transformers:    TRAINABLE  ({VideoViT.NUM_TEMPORAL_HEADS} heads)\n"
        f"  fusion_classifier:        TRAINABLE  (4096 temporal + 2 mean frame logits)\n"
        f"  Trainable params: {trainable_n:,} / {total_n:,} "
        f"({100*trainable_n/total_n:.1f}%)\n"
    )

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── AMP scaler ──────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── Optimiser & cosine scheduler with linear warmup ─────────────────────
    lr_base        = args.lr
    epochs         = args.epochs
    iter_per_epoch = len(train_loader)
    total_steps    = epochs * iter_per_epoch
    warmup_steps   = args.warmup_steps
    lr_min         = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmup_steps) * math.pi / (total_steps - warmup_steps))) / 2)
             + lr_min)
            if i > warmup_steps
            else (i / warmup_steps + lr_min)
        )
        for i in range(total_steps)
    }

    optimizer = optim.AdamW(trainable, lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[step]
    )

    lam = args.supcon_weight

    # ── Training loop ───────────────────────────────────────────────────────
    best_test_auc = 0.0
    best_epoch    = -1

    for epoch in range(epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)

        model.train()
        # Enforce eval mode on the frozen backbone so Dropout / BN use
        # inference statistics, not batch statistics.
        raw_for_eval = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_for_eval.frame_model.eval()

        iter_i = epoch * iter_per_epoch
        train_frame_labels, train_frame_probs = [], []
        train_video_labels = []
        train_video_mean_probs, train_video_with_frame_probs, train_video_no_frame_probs = [], [], []

        for batch_idx, (frames, labels, lengths) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        ):
            frames  = frames.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                (
                    video_logits_with_frame,
                    video_logits_no_frame,
                    frame_logits_list,
                    _,
                    video_feats_list,
                ) = model(frames, lengths)
                loss = stage2_loss(video_logits_with_frame, video_feats_list, labels, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                fl, fp, vl, vmp, vwfp, vnfp = _video_probs_from_forward(
                    video_logits_with_frame,
                    video_logits_no_frame,
                    frame_logits_list,
                    labels,
                    lengths,
                )
            train_frame_labels.extend(fl.cpu().numpy().tolist())
            train_frame_probs.extend(fp.cpu().numpy().tolist())
            train_video_labels.extend(vl.cpu().numpy().tolist())
            train_video_mean_probs.extend(vmp.cpu().numpy().tolist())
            train_video_with_frame_probs.extend(vwfp.cpu().numpy().tolist())
            train_video_no_frame_probs.extend(vnfp.cpu().numpy().tolist())

            if batch_idx % 256 == 0:
                print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

        print()
        compute_metrics(train_frame_labels,   train_frame_probs,     "Train (A) frame    ", epoch)
        compute_metrics(train_video_labels,   train_video_mean_probs, "Train (B) vid-mean ", epoch)
        compute_metrics(train_video_labels, train_video_with_frame_probs,
                        "Train (C) frame-end", epoch)
        compute_metrics(train_video_labels, train_video_no_frame_probs,
                        "Train (D) no-frame ", epoch)

        val_fl, val_fp, val_vl, val_vmp, val_vwfp, val_vnfp = run_eval(
            model, val_loader, f"Epoch {epoch+1} [val]"
        )
        compute_metrics(val_fl,  val_fp,  "Val   (A) frame    ", epoch)
        compute_metrics(val_vl,  val_vmp, "Val   (B) vid-mean ", epoch)
        val_with_frame_auc = compute_metrics(val_vl, val_vwfp,
                                             "Val   (C) frame-end", epoch)
        val_no_frame_auc = compute_metrics(val_vl, val_vnfp,
                                           "Val   (D) no-frame ", epoch)
        print(
            f"  [Val AUC summary] frame-end={val_with_frame_auc:.4f}  "
            f"no-frame={val_no_frame_auc:.4f}"
        )
        write_grid_metric_rows(epoch, "val", {
            "frame_end": val_with_frame_auc,
            "no_frame": val_no_frame_auc,
        })

        cdf_fl, cdf_fp, cdf_vl, cdf_vmp, cdf_vwfp, cdf_vnfp = run_eval(
            model, cdf_loader, f"Epoch {epoch+1} [CDFv1]"
        )
        compute_metrics(cdf_fl,  cdf_fp,  "Test  (A) frame    ", epoch)
        compute_metrics(cdf_vl,  cdf_vmp, "Test  (B) vid-mean ", epoch)
        test_auc = compute_metrics(cdf_vl, cdf_vwfp,
                                   "Test  (C) frame-end", epoch)
        test_no_frame_auc = compute_metrics(cdf_vl, cdf_vnfp,
                                            "Test  (D) no-frame ", epoch)
        print(
            f"  [Test AUC summary] frame-end={test_auc:.4f}  "
            f"no-frame={test_no_frame_auc:.4f}"
        )
        write_grid_metric_rows(epoch, "test", {
            "frame_end": test_auc,
            "no_frame": test_no_frame_auc,
        })
        # ── Checkpointing ────────────────────────────────────────────────────
        raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = raw_model.state_dict()

        torch.save(state_dict, os.path.join(args.save_root, "latest.pth"))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch    = epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            print(f"\n  ★ New best Test video AUC={best_test_auc:.4f} → saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test frame-end AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Stage 2 complete.")
    print(f"  Best checkpoint: epoch {best_epoch+1}  Test frame-end AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
    print(SEP)
