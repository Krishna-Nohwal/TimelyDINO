"""
tsne_predictions.py — UMAP visualization of VideoViT video-level embeddings
                       and predictions across FF++, CDFv2, WDF, UADFV, and DFo.

What this visualizes
---------------------
For every video in the selected evaluation datasets, this script runs the FULL
Stage-2 VideoViT forward pass (frame_model -> temporal transformers ->
fusion_classifier) and extracts:

  * temporal_vec       : (4096,) — concat of the 4 temporal-transformer head
                          outputs. This is exactly what the fusion_classifier
                          receives (modulo the 2 appended frame-mean-logit
                          dims), i.e. the model's internal "prediction
                          representation" for the video.
  * video_prob         : P(fake) from the frame-end video classifier (stream C
                          in the eval scripts — "frame-end w/ mem" style path).
  * video_label        : ground truth (0 = Real, 1 = Fake).
  * dataset            : 'FFPP' | 'CDFv2' | 'WDF' | 'UADFV' | 'DFo'.

All embeddings across the datasets are stacked, jointly reduced to 2D with a
single UMAP fit (so the datasets sit in one shared space), and plotted in
several paper-friendly views:
dataset+label, label-only, predicted P(fake), correctness, and real/fake split.

Model / architecture notes (see video_model.py and train_stage2_frame_end.py)
----------------------------------------------------------------------------
  - Backbone: frame_model.ViT (frame_model_4layers), ViT-Large, EMBED_DIM=1024,
    taps 4 transformer layers [20,21,22,23]; forward() -> 3-tuple
    (logits_list, features_list, cls_list). cls_list[i] is (B*T, 1024), the
    already-squeezed CLS token ("f_cls") for tapped layer i.
  - VideoViT (frame-end variant, matches train_stage2_frame_end.py):
        temporal_transformers : 4 x TemporalTransformer, one per tapped layer
        fusion_classifier      : Linear(4*1024 + 2, 2)
            input = concat(temporal_vec [4096], frame_mean_logits [2])
            frame_mean_logits = deepest SpatialHead (index 3) logits, averaged
            over valid frames -- a "shortcut" appended at the very end.
  - use_memory_bank (optional real-video kNN gate) is auto-detected from the
    checkpoint by checking for the 'memory_gate' key; the memory bank must be
    rebuilt from real training frames because the frame model is frozen.

Directory / manifest layout (from the uploaded scripts)
---------------------------------------------------------
  FF++ (train_stage2_frame_end.py):
      --manifest   CSV with columns {sample_dir, label}, label: 0=Real,1=Fake
      --root_dir   root such that <root_dir>/<sample_dir>/image.png exists
      A video-level 5% val split (seed=42) is carved out of this manifest via
      the exact same logic as prepare_splits() -- we visualize the VAL split
      only, so we are not just showing memorized training videos.

  CDFv2 (cdfv2_knn42.py):
      --fake_root  <fake_root>/<sample_dir>/image.png   (label=1, fake)
      --real_root  <real_root>/<sample_dir>/image.png   (label=0, real)
      sample_dir encodes video via trailing _fNNNN or _frame_NN suffix.

  DFo / UADFV:
      --df0_fake_root / --uadfv_fake_root  <root>/<video_id>/<frame_idx>/image.png
      --df0_real_root / --uadfv_real_root  <root>/<video_id>/<frame_idx>/image.png

  WDF:
      --wdf_fake_root / --wdf_real_root  <root>/<video_id>_<frame_number>.png

Full commands
-------------
First cache embeddings once:

python tsne_predictions1.py \
    --checkpoint /home/tarun/Desktop/best/best.pth \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/ \
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --df0_fake_root /media/tarun/B482367C823642E2/usr/df1.0_faces/fake \
    --df0_real_root /media/tarun/B482367C823642E2/usr/df1.0_faces/real \
    --wdf_fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --wdf_real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --uadfv_fake_root /media/tarun/B482367C823642E2/usr/uadfv_faces/fake \
    --uadfv_real_root /media/tarun/B482367C823642E2/usr/uadfv_faces/real \
    --num_frames 32 \
    --max_videos_per_dataset 0 \
    --outlier_std 2.0 \
    --outlier_group class \
    --train_real_root /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real \
    --embeddings_out cached_video_embeddings_ffpp_cdfv2_wdf_uadfv_df0.npz \
    --out umap_ffpp_cdfv2_wdf_uadfv_df0.png

Then reuse cached embeddings for plotting:

python tsne_predictions1.py \
    --embeddings_npz cached_video_embeddings_ffpp_cdfv2_wdf_uadfv_df0.npz \
    --out umap_ffpp_cdfv2_wdf_uadfv_df0.png

If --embeddings_out is omitted, embeddings are automatically cached next to the
figure as <out_stem>_embeddings.npz before UMAP is imported.
If --embeddings_out points to an existing file, that cache is reused unless
--force_extract is supplied.

Any of --manifest/--root_dir, --cdfv2_fake_root/--cdfv2_real_root,
--df0_fake_root/--df0_real_root, --wdf_fake_root/--wdf_real_root, or
--uadfv_fake_root/--uadfv_real_root may be omitted to skip that dataset.
"""

import argparse
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from augmentations import load_and_resize, normalize
from frame_model import ViT
from video_model import RealVideoMemoryBank, TemporalTransformer


# ---------------------------------------------------------------------------
# VideoViT (frame-end variant) -- identical architecture to
# train_stage2_frame_end.py so a Stage-2 checkpoint loads with strict=True.
# ---------------------------------------------------------------------------

class VideoViT(nn.Module):
    """Stage 2 frame-end video model (see module docstring)."""

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
        self.memory_bank: Optional[RealVideoMemoryBank] = None
        self.memory_gate = nn.Parameter(
            torch.full((self.NUM_TEMPORAL_HEADS, 1, 1), -2.0)
        ) if use_memory_bank else None

        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim=ViT.EMBED_DIM,
                num_frames=num_frames,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])
        self.fusion_classifier = nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM + 2, 2
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

    def forward(self, video: Tensor, lengths: Optional[Tensor] = None):
        """
        Returns
        -------
        video_logits_with_frame : (B, 2) — primary Stage-2 video prediction
        video_logits_no_frame   : (B, 2) — ablation w/o frame-logit shortcut
        temporal_vec            : (B, 4096) — the embedding we visualize
        """
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

        temporal_vec = torch.cat(video_feats_list, dim=1)  # (B, 4096)
        frame_mean_logits = self._mean_valid_frame_logits(
            frame_logits_list, B, T, key_padding_mask, temporal_vec.dtype
        )

        fused_with_frame = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        fused_no_frame = torch.cat(
            [temporal_vec, torch.zeros_like(frame_mean_logits)], dim=1
        )

        video_logits_with_frame = self.fusion_classifier(fused_with_frame)
        video_logits_no_frame = self.fusion_classifier(fused_no_frame)

        return video_logits_with_frame, video_logits_no_frame, temporal_vec


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, num_frames: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}

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
            f"Expected {expected_fusion_dim} for the frame-end architecture "
            f"({VideoViT.NUM_TEMPORAL_HEADS}*{VideoViT.EMBED_DIM}+2). "
            f"In this architecture the memory bank gates CLS tokens BEFORE "
            f"the temporal transformers, so fusion_classifier's input dim "
            f"does not change whether or not a memory bank was used -- "
            f"the 'memory_gate' key presence is the correct signal instead."
        )

    # The memory bank (if used) gates CLS token sequences before the temporal
    # transformers; it adds a 'memory_gate' parameter but does NOT change
    # fusion_classifier's input dimensionality. So detect use_memory_bank from
    # key presence, not from the fusion layer's shape.
    use_memory_bank = "memory_gate" in ckpt
    print(f"  fusion_classifier input dim={fusion_in_dim}  "
          f"memory_gate present={use_memory_bank} -> use_memory_bank={use_memory_bank}")

    model = VideoViT(num_frames=num_frames, use_memory_bank=use_memory_bank).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("  Checkpoint loaded successfully.")
    return model, use_memory_bank


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 256


def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def _load_frames(paths: List[str], img_size: int = IMG_SIZE) -> torch.Tensor:
    frames = []
    for p in paths:
        try:
            img = load_and_resize(p, img_size)
            img = normalize(img)
        except Exception:
            img = torch.zeros(3, img_size, img_size)
        frames.append(img)
    return torch.stack(frames, dim=0)


def clip_collate_fn(batch):
    """batch items: (frames, label, video_id, dataset_name)"""
    frames_list, labels, vids, dsets = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        padded.append(F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t)) if pad_t > 0 else f)
    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
        list(vids),
        list(dsets),
    )


def video_id_from_sample(sample_name: str) -> str:
    """CDFv2 / FF++ style: strip trailing _fNNNN or _frame_NN suffix."""
    return re.sub(r"_(?:frame_|f)\d+$", "", sample_name)


# ---------------------------------------------------------------------------
# Generic clip dataset — one entry per video, uniform-stride frame sampling.
# Each dataset builder below produces a list[(video_id, [frame_paths], label)]
# which is fed into this single Dataset class, tagged with a dataset name.
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    def __init__(self, videos: List[Tuple[str, List[str], int]], dataset_name: str,
                 num_frames: int):
        self.videos = videos
        self.dataset_name = dataset_name
        self.num_frames = num_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = _sample_frame_indices(len(paths), self.num_frames)
        sampled = [paths[i] for i in indices]
        frames = _load_frames(sampled)
        return frames, label, vid, self.dataset_name


# ---------------------------------------------------------------------------
# Dataset-specific video-list builders
# ---------------------------------------------------------------------------

def _extract_video_id_ffpp(sample_dir: str) -> str:
    """Same logic as train_stage2_frame_end.py's _extract_video_id."""
    parts = Path(sample_dir).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = basename.rfind("_frame_")
    if idx != -1:
        clip_id = basename[:idx]
    else:
        m = re.search(r"_f\d+$", basename)
        clip_id = basename[:m.start()] if m else basename
    return f"{prefix}/{clip_id}" if prefix else clip_id


def build_ffpp_videos(
    manifest_csv: str, root_dir: str, val_ratio: float = 0.05
) -> List[Tuple[str, List[str], int]]:
    """
    Rebuilds the exact video-level val split used by train_stage2_frame_end.py
    (prepare_splits, seed=42) and returns only the val videos, so we visualize
    held-out FF++ videos rather than ones the model trained on directly.
    """
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")

    df["video_id"] = df["sample_dir"].apply(_extract_video_id_ffpp)

    real_vids = df[df["label"] == 0]["video_id"].unique()
    fake_vids = df[df["label"] == 1]["video_id"].unique()

    rng = np.random.default_rng(42)
    real_vids = rng.permutation(real_vids)
    fake_vids = rng.permutation(fake_vids)

    real_val_n = max(1, int(len(real_vids) * val_ratio))
    fake_val_n = max(1, int(len(fake_vids) * val_ratio))
    val_ids = set(real_vids[:real_val_n]) | set(fake_vids[:fake_val_n])

    val_df = df[df["video_id"].isin(val_ids)].reset_index(drop=True)

    root = Path(root_dir)
    videos = []
    for video_id, group in val_df.groupby("video_id"):
        label = int(group["label"].iloc[0])
        paths = []
        for rel in group["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
            img_path = root / rel / "image.png"
            if img_path.is_file():
                paths.append(str(img_path))
        paths = sorted(paths)
        if paths:
            videos.append((video_id, paths, label))

    real_n = sum(1 for _, _, l in videos if l == 0)
    fake_n = sum(1 for _, _, l in videos if l == 1)
    print(f"  [FF++ val split] {len(videos)} videos  (real={real_n}, fake={fake_n})")
    return videos


def build_cdfv2_videos(fake_root: str, real_root: str) -> List[Tuple[str, List[str], int]]:
    fake_root, real_root = Path(fake_root), Path(real_root)
    vid2paths: dict = defaultdict(list)
    vid2label: dict = {}

    for root, label in [(fake_root, 1), (real_root, 0)]:
        if not root.is_dir():
            print(f"  [CDFv2] WARNING: {root} does not exist, skipping.")
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "image.png").exists():
                vid = video_id_from_sample(d.name)
                vid2paths[vid].append(str(d / "image.png"))
                vid2label[vid] = label

    videos = []
    for vid, paths in sorted(vid2paths.items()):
        videos.append((vid, sorted(paths), vid2label[vid]))

    real_n = sum(1 for _, _, l in videos if l == 0)
    fake_n = sum(1 for _, _, l in videos if l == 1)
    print(f"  [CDFv2] {len(videos)} videos  (real={real_n}, fake={fake_n})")
    return videos


def _sort_nested_frame_paths(paths: List[Path]) -> List[str]:
    def key_fn(path: Path):
        parent = path.parent.name
        if parent.isdigit():
            return int(parent), str(path)
        return 10**12, str(path)

    return [str(path) for path in sorted(paths, key=key_fn)]


def build_nested_image_videos(
    fake_root: str,
    real_root: str,
    dataset_name: str,
) -> List[Tuple[str, List[str], int]]:
    """
    Builds videos for DFo/UADFV style directories:
        <fake_root>/<video_id>/<frame_idx>/image.png
        <real_root>/<video_id>/<frame_idx>/image.png
    """
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: {root} does not exist, skipping.")
            continue
        for video_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            paths = _sort_nested_frame_paths(list(video_dir.rglob("image.png")))
            if paths:
                videos.append((video_dir.name, paths, label))

    real_n = sum(1 for _, _, l in videos if l == 0)
    fake_n = sum(1 for _, _, l in videos if l == 1)
    print(f"  [{dataset_name}] {len(videos)} videos  (real={real_n}, fake={fake_n})")
    return videos


_FLAT_FRAME_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def build_wdf_videos(fake_root: str, real_root: str) -> List[Tuple[str, List[str], int]]:
    """
    Builds videos for WDF's flat layout:
        <root>/<video_id>_<frame_number>.png
    """
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [WDF] WARNING: {root} does not exist, skipping.")
            continue

        grouped = defaultdict(list)
        skipped = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = _FLAT_FRAME_RE.match(path.name)
            if not match:
                skipped += 1
                continue
            video_id, frame_idx = match.group(1), int(match.group(2))
            grouped[video_id].append((frame_idx, str(path)))

        if skipped:
            print(f"  [WDF] skipped {skipped} files under {root}")

        for video_id, indexed_paths in sorted(grouped.items()):
            paths = [p for _, p in sorted(indexed_paths)]
            if paths:
                videos.append((video_id, paths, label))

    real_n = sum(1 for _, _, l in videos if l == 0)
    fake_n = sum(1 for _, _, l in videos if l == 1)
    print(f"  [WDF] {len(videos)} videos  (real={real_n}, fake={fake_n})")
    return videos


# ---------------------------------------------------------------------------
# Real-video memory bank rebuild (only needed if checkpoint used it)
# ---------------------------------------------------------------------------

class _RealClipDataset(Dataset):
    """Loads real training clips for kNN bank construction. No augmentation."""

    def __init__(self, real_train_root: Path, num_frames: int):
        self.num_frames = num_frames
        subdirs = sorted(d for d in real_train_root.iterdir() if d.is_dir())
        is_flat = any(re.search(r"_(?:frame_|f)\d+$", d.name) for d in subdirs)

        self.videos: list = []
        if is_flat:
            vid2paths: dict = defaultdict(list)
            for d in subdirs:
                img_path = d / "image.png"
                if img_path.exists():
                    vid = re.sub(r"_(?:frame_|f)\d+$", "", d.name)
                    vid2paths[vid].append(str(img_path))
            for vid, paths in sorted(vid2paths.items()):
                self.videos.append((vid, sorted(paths)))
        else:
            for video_dir in subdirs:
                paths = sorted(str(p) for p in video_dir.rglob("image.png"))
                if paths:
                    self.videos.append((video_dir.name, paths))

        print(f"  [RealVideoBank] {len(self.videos)} real training videos "
              f"({'flat' if is_flat else 'nested'} layout)")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths = self.videos[idx]
        indices = _sample_frame_indices(len(paths), self.num_frames)
        frames = _load_frames([paths[i] for i in indices])
        return frames, 0


def _real_clip_collate(batch):
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        padded.append(F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t)) if pad_t > 0 else f)
    return torch.stack(padded, dim=0), torch.tensor(labels, dtype=torch.long), lengths


def build_memory_bank(
    model: VideoViT, real_train_root: str, num_frames: int, k: int,
    batch_size: int, num_workers: int, device: torch.device,
) -> RealVideoMemoryBank:
    print("\n  Building real-video kNN memory bank ...")
    dataset = _RealClipDataset(Path(real_train_root), num_frames=num_frames)
    if len(dataset) == 0:
        raise RuntimeError(f"No real training videos found under {real_train_root}.")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_real_clip_collate,
    )
    bank = RealVideoMemoryBank(
        embed_dim=VideoViT.EMBED_DIM, num_heads=VideoViT.NUM_TEMPORAL_HEADS, k=k,
    )

    was_training = model.training
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, _, lengths in tqdm(loader, desc="  Building bank", leave=False):
            B, T, C, H, W = frames.shape
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            flat_frames = frames.reshape(B * T, C, H, W)
            _, _, cls_list = model.frame_model(flat_frames)

            time_idx = torch.arange(T, device=device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.unsqueeze(1)

            cls_sequences = [
                cls_tokens.reshape(B, T, model.EMBED_DIM) for cls_tokens in cls_list
            ]
            bank.add(cls_sequences, key_padding_mask)

    bank.build()
    if was_training:
        model.train()
    print(f"  Memory bank ready: {len(bank)} real-video CLS prototypes, k={k}")
    return bank


# ---------------------------------------------------------------------------
# Inference: extract temporal_vec embeddings + predictions for a dataset
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_embeddings(
    model: VideoViT, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Returns
    -------
    embeddings  : (N, 4096) float32   -- temporal_vec per video
    labels      : (N,) int            -- ground truth, 0=Real, 1=Fake
    probs       : (N,) float          -- P(fake) from frame-end classifier
    video_ids   : list[str]
    dataset_tags: list[str]
    """
    model.eval()
    all_embeds, all_labels, all_probs = [], [], []
    all_vids, all_dsets = [], []

    with torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, labels, lengths, vids, dsets in tqdm(loader, desc="Extracting", leave=False):
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            video_logits_with_frame, _, temporal_vec = model(frames, lengths)
            probs = torch.softmax(video_logits_with_frame.float(), dim=1)[:, 1]

            all_embeds.append(temporal_vec.float().cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_vids.extend(vids)
            all_dsets.extend(dsets)

    embeddings = np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, VideoViT.NUM_TEMPORAL_HEADS * VideoViT.EMBED_DIM))
    return embeddings, np.array(all_labels), np.array(all_probs), all_vids, all_dsets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="UMAP of VideoViT video-level embeddings/predictions "
                    "across FF++, CDFv2, WDF, UADFV, and DFo."
    )
    p.add_argument("--checkpoint", default="",
                   help="Path to a Stage-2 frame-end VideoViT checkpoint (.pth). "
                        "Not needed if --embeddings_npz is supplied.")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_videos_per_dataset", type=int, default=0,
                   help="Optional cap per dataset. 0 = use all available videos "
                        "from every dataset.")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--out", default="umap_predictions.png",
                   help="Main all-in-one UMAP plot path. Extra views are saved "
                        "next to it with suffixes.")
    p.add_argument("--embeddings_out", default="",
                   help="Optional .npz path to cache raw embeddings/labels/probs "
                        "(useful to re-plot without rerunning the model). If this "
                        "file already exists, it is loaded unless --force_extract "
                        "is set.")
    p.add_argument("--embeddings_npz", default="",
                   help="Optional cached .npz containing embeddings, labels, probs, "
                        "video_ids, and dataset_tags. If supplied, model inference "
                        "and dataset loading are skipped.")
    p.add_argument("--force_extract", action="store_true",
                   help="Recompute embeddings even if --embeddings_out already exists.")
    p.add_argument("--outlier_std", type=float, default=2.0,
                   help="Remove embedding outliers farther than mean + N*std "
                        "within each outlier group. Default 2.0 is stronger. "
                        "Set <= 0 to disable.")
    p.add_argument("--outlier_group", default="class",
                   choices=["class", "dataset_class", "dataset", "all"],
                   help="Grouping used to compute outlier thresholds.")

    # FF++
    p.add_argument("--manifest", default="",
                   help="FF++ manifest CSV (train_stage2_frame_end.py format).")
    p.add_argument("--root_dir", default="",
                   help="FF++ frame root dir.")
    p.add_argument("--val_ratio", type=float, default=0.05)

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # DFo / DeeperForensics-1.0 style nested roots.
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")

    # WDF flat roots
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV style nested roots
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")

    # Memory bank (only needed if checkpoint has one)
    p.add_argument("--train_real_root", default="",
                   help="Real training frames root, required if the checkpoint "
                        "was trained with --use_memory_bank.")
    p.add_argument("--knn_k", type=int, default=32)
    p.add_argument("--bank_batch_size", type=int, default=16)

    # UMAP params
    p.add_argument("--umap_neighbors", type=int, default=50)
    p.add_argument("--umap_min_dist", type=float, default=0.02)
    p.add_argument("--umap_metric", default="cosine")
    p.add_argument("--umap_seed", type=int, default=42)

    return p.parse_args()


def _stratified_cap(videos: List[Tuple[str, List[str], int]], cap: int, seed: int = 0):
    if cap <= 0 or len(videos) <= cap:
        return videos
    rng = np.random.default_rng(seed)
    reals = [v for v in videos if v[2] == 0]
    fakes = [v for v in videos if v[2] == 1]
    n_real = min(len(reals), cap // 2)
    n_fake = min(len(fakes), cap - n_real)
    n_real = min(len(reals), cap - n_fake)
    reals = [reals[i] for i in rng.choice(len(reals), size=n_real, replace=False)] if reals else []
    fakes = [fakes[i] for i in rng.choice(len(fakes), size=n_fake, replace=False)] if fakes else []
    return reals + fakes


def _dataset_sampled_items(
    dataset_videos: dict,
    videos_per_dataset: int,
) -> List[Tuple[str, List[str], int, str]]:
    available = {name: len(videos) for name, videos in dataset_videos.items() if videos}
    if not available:
        return []

    print("\n  Dataset sampling:")
    for name, count in available.items():
        sampled_count = count if videos_per_dataset <= 0 else min(videos_per_dataset, count)
        print(f"    {name}: available={count}  sampled={sampled_count}")

    sampled_items = []
    for seed, (name, videos) in enumerate(sorted(dataset_videos.items()), start=10):
        sampled = _stratified_cap(videos, videos_per_dataset, seed=seed)
        rng = np.random.default_rng(seed + 1000)
        order = rng.permutation(len(sampled))
        sampled = [sampled[i] for i in order]
        n_real = sum(1 for _, _, label in sampled if label == 0)
        n_fake = sum(1 for _, _, label in sampled if label == 1)
        print(f"    {name}: final real={n_real}  fake={n_fake}")
        sampled_items.extend((vid, paths, label, name) for vid, paths, label in sampled)
    return sampled_items


def _output_with_suffix(out_path: str, suffix: str) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def _set_tight_limits(ax, points: np.ndarray, pad_frac: float = 0.06):
    if points.size == 0:
        return
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    ax.set_xlim(x_min - pad_frac * x_span, x_max + pad_frac * x_span)
    ax.set_ylim(y_min - pad_frac * y_span, y_max + pad_frac * y_span)


def _default_embeddings_path(out_path: str) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_embeddings.npz"))


def _load_cached_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"].astype(int)
    probs = data["probs"]
    video_ids = data["video_ids"].astype(str).tolist()
    dataset_tags = data["dataset_tags"].astype(str).tolist()
    print(f"  Loaded cached embeddings: {embeddings.shape} from {path}")
    return embeddings, labels, probs, video_ids, dataset_tags


def _print_dataset_counts(title: str, labels: np.ndarray, dataset_tags: List[str]):
    tags = np.asarray(dataset_tags)
    print(f"\n  {title}")
    print("    dataset      total   real   fake")
    print("    -------------------------------")
    for dset in sorted(set(tags.tolist())):
        mask = tags == dset
        real_n = int(((labels == 0) & mask).sum())
        fake_n = int(((labels == 1) & mask).sum())
        print(f"    {dset:<10} {int(mask.sum()):>5} {real_n:>6} {fake_n:>6}")
    print(f"    {'TOTAL':<10} {len(labels):>5} {int((labels == 0).sum()):>6} {int((labels == 1).sum()):>6}")


def _outlier_group_keys(labels: np.ndarray, dataset_tags: List[str], mode: str):
    tags = np.asarray(dataset_tags)
    if mode == "class":
        return np.asarray([f"label={label}" for label in labels])
    if mode == "dataset":
        return tags.astype(str)
    if mode == "dataset_class":
        return np.asarray([f"{dset}/label={label}" for dset, label in zip(tags, labels)])
    return np.asarray(["all"] * len(labels))


def _remove_embedding_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    video_ids: List[str],
    dataset_tags: List[str],
    std_factor: float,
    group_mode: str,
):
    if std_factor <= 0:
        print("\n  Outlier removal disabled (--outlier_std <= 0).")
        return embeddings, labels, probs, video_ids, dataset_tags

    keys = _outlier_group_keys(labels, dataset_tags, group_mode)
    keep = np.ones(len(labels), dtype=bool)

    print(f"\n  Outlier removal: group={group_mode}, threshold=mean+{std_factor:.2f}*std")
    for key in sorted(set(keys.tolist())):
        idx = np.where(keys == key)[0]
        if len(idx) < 4:
            print(f"    {key:<20} n={len(idx):>5}  removed=0  (too small)")
            continue
        group_emb = embeddings[idx].astype(np.float64)
        centroid = group_emb.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(group_emb - centroid, axis=1)
        threshold = distances.mean() + std_factor * distances.std()
        group_keep = distances <= threshold
        keep[idx] = group_keep
        print(
            f"    {key:<20} n={len(idx):>5}  removed={int((~group_keep).sum()):>4}  "
            f"thr={threshold:.4f}"
        )

    removed = int((~keep).sum())
    print(f"  Total outliers removed: {removed}/{len(labels)}")
    if removed == 0:
        return embeddings, labels, probs, video_ids, dataset_tags

    filtered_video_ids = [vid for vid, is_keep in zip(video_ids, keep) if is_keep]
    filtered_dataset_tags = [tag for tag, is_keep in zip(dataset_tags, keep) if is_keep]
    return (
        embeddings[keep],
        labels[keep],
        probs[keep],
        filtered_video_ids,
        filtered_dataset_tags,
    )


def _patch_coverage_for_numba():
    """
    Some numba/coverage version pairs fail during import because numba expects
    coverage.types.Tracer, while newer coverage exposes TTracer. Patch the alias
    before importing umap/numba so plotting still works on that environment.
    """
    try:
        import coverage
    except Exception:
        return

    coverage_types = getattr(coverage, "types", None)
    if coverage_types is None:
        return
    if not hasattr(coverage_types, "Tracer") and hasattr(coverage_types, "TTracer"):
        coverage_types.Tracer = coverage_types.TTracer


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    sep = "=" * 78
    print(f"\n{sep}\n  UMAP of VideoViT predictions — cross-dataset video embeddings\n{sep}")
    print(f"  Device     : {device}")
    if args.embeddings_npz:
        embeddings, labels, probs, video_ids, dataset_tags = _load_cached_embeddings(
            args.embeddings_npz
        )
    elif args.embeddings_out and Path(args.embeddings_out).is_file() and not args.force_extract:
        embeddings, labels, probs, video_ids, dataset_tags = _load_cached_embeddings(
            args.embeddings_out
        )
    else:
        if not args.checkpoint:
            raise ValueError("Supply --checkpoint or --embeddings_npz.")
        print(f"  Checkpoint : {args.checkpoint}")

        # ---- Model ---------------------------------------------------------
        model, use_memory_bank = load_model(args.checkpoint, args.num_frames, device)

        if use_memory_bank:
            if not args.train_real_root:
                raise ValueError(
                    "Checkpoint uses a memory bank (use_memory_bank=True); "
                    "please supply --train_real_root."
                )
            bank = build_memory_bank(
                model, args.train_real_root, args.num_frames,
                args.knn_k, args.bank_batch_size, args.num_workers, device,
            )
            model.attach_memory_bank(bank)

        if not args.no_compile and hasattr(torch, "compile"):
            print("  Compiling model with torch.compile ...")
            model = torch.compile(model)

        # ---- Build video lists per dataset ---------------------------------
        dataset_videos = {}

        if args.manifest and args.root_dir:
            print("\n  Building FF++ video list (val split only) ...")
            dataset_videos["FFPP"] = build_ffpp_videos(
                args.manifest, args.root_dir, args.val_ratio
            )
        else:
            print("\n  [skip] FF++: --manifest / --root_dir not provided.")

        if args.cdfv2_fake_root and args.cdfv2_real_root:
            print("\n  Building CDFv2 video list ...")
            dataset_videos["CDFv2"] = build_cdfv2_videos(
                args.cdfv2_fake_root, args.cdfv2_real_root
            )
        else:
            print("\n  [skip] CDFv2: --cdfv2_fake_root / --cdfv2_real_root not provided.")

        if args.df0_fake_root and args.df0_real_root:
            print("\n  Building DFo video list ...")
            dataset_videos["DFo"] = build_nested_image_videos(
                args.df0_fake_root, args.df0_real_root, "DFo"
            )
        else:
            print("\n  [skip] DFo: --df0_fake_root / --df0_real_root not provided.")

        if args.wdf_fake_root and args.wdf_real_root:
            print("\n  Building WDF video list ...")
            dataset_videos["WDF"] = build_wdf_videos(
                args.wdf_fake_root, args.wdf_real_root
            )
        else:
            print("\n  [skip] WDF: --wdf_fake_root / --wdf_real_root not provided.")

        if args.uadfv_fake_root and args.uadfv_real_root:
            print("\n  Building UADFV video list ...")
            dataset_videos["UADFV"] = build_nested_image_videos(
                args.uadfv_fake_root, args.uadfv_real_root, "UADFV"
            )
        else:
            print("\n  [skip] UADFV: --uadfv_fake_root / --uadfv_real_root not provided.")

        all_videos: List[Tuple[str, List[str], int, str]] = _dataset_sampled_items(
            dataset_videos, args.max_videos_per_dataset
        )

        if not all_videos:
            raise ValueError(
                "No datasets were provided. Supply at least one dataset root/manifest pair."
            )

        print(f"\n  Total videos across all datasets: {len(all_videos)}")

        # ---- Dataset / loader ----------------------------------------------
        class _CombinedDataset(Dataset):
            def __init__(self, items):
                self.items = items
                self.num_frames = args.num_frames

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                vid, paths, label, dset = self.items[idx]
                indices = _sample_frame_indices(len(paths), self.num_frames)
                frames = _load_frames([paths[i] for i in indices])
                return frames, label, vid, dset

        combined_dataset = _CombinedDataset(all_videos)
        loader = DataLoader(
            combined_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=clip_collate_fn,
        )

        # ---- Run model, extract embeddings ---------------------------------
        print("\n  Running VideoViT forward passes to extract embeddings ...")
        embeddings, labels, probs, video_ids, dataset_tags = extract_embeddings(model, loader, device)
        print(f"  Extracted embeddings: {embeddings.shape}")

        embeddings_out = args.embeddings_out or _default_embeddings_path(args.out)
        np.savez(
            embeddings_out,
            embeddings=embeddings, labels=labels, probs=probs,
            video_ids=np.array(video_ids), dataset_tags=np.array(dataset_tags),
        )
        print(f"  Cached raw embeddings -> {embeddings_out}")

    _print_dataset_counts("Videos loaded before outlier removal:", labels, dataset_tags)
    embeddings, labels, probs, video_ids, dataset_tags = _remove_embedding_outliers(
        embeddings,
        labels,
        probs,
        video_ids,
        dataset_tags,
        args.outlier_std,
        args.outlier_group,
    )
    _print_dataset_counts("Videos used for UMAP after outlier removal:", labels, dataset_tags)

    # ---- UMAP ----------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
    _patch_coverage_for_numba()
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requires the 'umap-learn' package. Install it with: "
            "pip install umap-learn"
        ) from exc
    except AttributeError as exc:
        raise RuntimeError(
            "UMAP import failed inside numba/coverage. The embeddings have already "
            "been cached, so rerun plotting with --embeddings_npz <cache>. If this "
            "persists, fix the environment with one of: "
            "pip install -U numba coverage umap-learn, or pip uninstall coverage."
        ) from exc

    print("\n  Standardizing embeddings and running joint UMAP ...")
    scaled = StandardScaler().fit_transform(embeddings)

    n = scaled.shape[0]
    n_neighbors = min(args.umap_neighbors, max(2, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
    )
    coords = reducer.fit_transform(scaled)

    # ---- Plot ------------------------------------------------------------------
    saved = plot_umap(coords, labels, probs, dataset_tags, args.out)
    print("\n  Saved plots:")
    for path in saved:
        print(f"    {path}")
    print(sep)


def plot_umap(coords: np.ndarray, labels: np.ndarray, probs: np.ndarray,
              dataset_tags: List[str], out_path: str):
    import matplotlib.pyplot as plt

    dataset_tags = np.array(dataset_tags)
    preferred_order = ["FFPP", "CDFv2", "WDF", "UADFV", "DFo"]
    present = set(dataset_tags.tolist())
    datasets = [d for d in preferred_order if d in present]
    datasets.extend(sorted(present - set(datasets)))
    base_colors = {
        "FFPP": "#0057FF",   # blue
        "CDFv2": "#E31A1C",  # red
        "WDF": "#000000",    # black
        "UADFV": "#FFD700",  # yellow
        "DFo": "#FF4FB3",    # pink
    }
    fallback_colors = plt.get_cmap("tab20").colors
    dataset_colors = {
        dset: base_colors.get(dset, fallback_colors[i % len(fallback_colors)])
        for i, dset in enumerate(datasets)
    }
    saved = []

    # -- Main: colored by dataset, marker by real/fake -----------------------
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))
    for dset in datasets:
        mask_d = dataset_tags == dset
        for label, marker, name in [(0, "o", "Real"), (1, "^", "Fake")]:
            mask = mask_d & (labels == label)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=dataset_colors[dset], marker=marker,
                s=10, alpha=0.82, linewidths=0.1, edgecolors="black",
                label=f"{dset} - {name}",
            )
    ax.set_title("UMAP of video embeddings\ncolor = dataset, shape = real/fake")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=8, loc="best", markerscale=1.2)
    _set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)

    # -- Label-only view ------------------------------------------------------
    label_path = _output_with_suffix(out_path, "label")
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))
    label_styles = {
        0: ("#2E7D32", "o", "Real"),
        1: ("#C62828", "^", "Fake"),
    }
    for label, (color, marker, name) in label_styles.items():
        mask = labels == label
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, marker=marker, s=10, alpha=0.82,
            linewidths=0.1, edgecolors="black", label=name,
        )
    ax.set_title("UMAP of video embeddings\ncolor = real/fake")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=9, loc="best")
    _set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(label_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(label_path)

    # -- Predicted probability view ------------------------------------------
    prob_path = _output_with_suffix(out_path, "prob")
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=probs, cmap="coolwarm",
        vmin=0, vmax=1, s=10, alpha=0.82, linewidths=0.1, edgecolors="black",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Predicted P(fake)")
    ax.set_title("UMAP of video embeddings\ncolor = model's predicted P(fake)")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    _set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(prob_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(prob_path)

    # -- Correctness view -----------------------------------------------------
    correct_path = _output_with_suffix(out_path, "correctness")
    preds = (probs >= 0.5).astype(int)
    correct = preds == labels
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))
    for mask, color, marker, name in [
        (correct, "#4C72B0", "o", "Correct"),
        (~correct, "#D62728", "x", "Wrong"),
    ]:
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, marker=marker, s=12, alpha=0.82,
            linewidths=0.55, edgecolors="black" if marker != "x" else color,
            label=name,
        )
    ax.set_title("UMAP of video embeddings\ncolor = prediction correctness")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=9, loc="best")
    _set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(correct_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(correct_path)

    # -- Split real/fake panels ----------------------------------------------
    split_path = _output_with_suffix(out_path, "real_fake_split")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, label, title in [(axes[0], 0, "Real videos"), (axes[1], 1, "Fake videos")]:
        panel_mask = labels == label
        for dset in datasets:
            mask = (labels == label) & (dataset_tags == dset)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=dataset_colors[dset], s=8, alpha=0.84,
                linewidths=0.08, edgecolors="black", label=dset,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP dim 1")
        ax.set_ylabel("UMAP dim 2")
        ax.legend(fontsize=8, loc="best")
        _set_tight_limits(ax, coords[panel_mask])
    fig.suptitle("UMAP split by class")
    fig.tight_layout()
    fig.savefig(split_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(split_path)

    return saved


if __name__ == "__main__":
    main()
