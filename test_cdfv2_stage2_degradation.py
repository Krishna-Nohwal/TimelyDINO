"""
Robustness test for the original Stage-2 model with RVMB on CDFv2.

Example
-------
python test_cdfv2_stage2_degradation.py \
    --checkpoint frame_end_mem_gate0p13/best.pth \
    --fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --train_real_root /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real \
    --out_dir cdfv2_degradation_stage2_mem \
    --num_frames 32 --batch_size 4 --num_workers 8 --no_compile

The real-video memory bank is built once from clean FF++ real training videos.
Only the CDFv2 test frames are degraded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
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
from video_model import RealVideoMemoryBank, TemporalTransformer

try:
    from augmentations import normalize as project_normalize
except Exception:
    project_normalize = None


IMG_SIZE = 256
DEEPEST_HEAD_IDX = ViT.NUM_LAYERS - 1
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


class VideoViT(torch.nn.Module):
    """Stage-2 frame-end VideoViT, matching cdfv2_knn42.py."""

    EMBED_DIM = ViT.EMBED_DIM
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
        use_memory_bank: bool = True,
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
                embed_dim=self.EMBED_DIM,
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
    def _mean_valid_frame_logits(frame_logits_list, bsz, time_steps, key_padding_mask, dtype):
        frame_logits = frame_logits_list[-1].float().reshape(bsz, time_steps, 2)
        if key_padding_mask is None:
            return frame_logits.mean(dim=1).to(dtype=dtype)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        counts = valid.sum(dim=1).clamp(min=1)
        return ((frame_logits * valid).sum(dim=1) / counts).to(dtype=dtype)

    def forward(self, video: torch.Tensor, lengths=None):
        bsz, time_steps, channels, height, width = video.shape
        if time_steps > self.num_frames:
            raise ValueError(f"Expected <= {self.num_frames} frames, got {time_steps}")

        frames = video.reshape(bsz * time_steps, channels, height, width)
        frame_logits_list, frame_feats_list, cls_list = self.frame_model(frames)

        if lengths is None:
            key_padding_mask = None
        else:
            time_idx = torch.arange(time_steps, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)

        cls_sequences = [
            cls_tokens.reshape(bsz, time_steps, self.EMBED_DIM) for cls_tokens in cls_list
        ]

        memory_refs = None
        if self.use_memory_bank:
            if self.memory_bank is None:
                raise RuntimeError("use_memory_bank=True but no bank is attached.")
            memory_refs = self.memory_bank.query(cls_sequences, key_padding_mask)

        video_feats_list = []
        for h, (temporal_tfm, frame_cls) in enumerate(
            zip(self.temporal_transformers, cls_sequences)
        ):
            if memory_refs is not None:
                gate = torch.sigmoid(self.memory_gate[h]).to(dtype=frame_cls.dtype)
                frame_cls = (1 - gate) * frame_cls + gate * memory_refs[h].unsqueeze(1)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)
        frame_mean_logits = self._mean_valid_frame_logits(
            frame_logits_list, bsz, time_steps, key_padding_mask, temporal_vec.dtype
        )
        fused_with_frame = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        fused_no_frame = torch.cat(
            [temporal_vec, torch.zeros_like(frame_mean_logits)], dim=1
        )

        return (
            self.fusion_classifier(fused_with_frame),
            self.fusion_classifier(fused_no_frame),
            frame_logits_list,
            frame_feats_list,
            video_feats_list,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CDFv2 Stage-2 RVMB robustness under image degradation."
    )
    parser.add_argument("--checkpoint", default="frame_end_mem_gate0p13/best.pth")
    parser.add_argument(
        "--fake_root",
        default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2",
    )
    parser.add_argument(
        "--real_root",
        default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real",
    )
    parser.add_argument(
        "--train_real_root",
        default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real",
        help="Clean FF++ real-frame root used to rebuild RVMB.",
    )
    parser.add_argument("--out_dir", default="cdfv2_degradation_stage2_mem")
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--bank_batch_size", default=16, type=int)
    parser.add_argument("--knn_k", default=32, type=int)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument(
        "--degradations",
        default="blur,jpeg,noise,downsample",
        help="Comma-separated subset of: blur,jpeg,noise,downsample.",
    )
    parser.add_argument("--blur_levels", default="0,1,2,4,6")
    parser.add_argument("--jpeg_levels", default="0,20,40,60,80,90")
    parser.add_argument("--noise_levels", default="0,0.02,0.04,0.08,0.12,0.16")
    parser.add_argument("--downsample_levels", default="1,2,4,8,12,16")
    parser.add_argument(
        "--save_per_video",
        action="store_true",
        help="Write one per-video CSV per degradation level.",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def video_id_from_sample(sample_name: str) -> str:
    import re

    return re.sub(r"_(?:frame_|f)\d+$", "", sample_name)


def sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def tensor_from_pil(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if project_normalize is not None:
        return project_normalize(tensor)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def deterministic_rng(path: str, degradation: str, severity: float) -> np.random.Generator:
    key = f"{path}|{degradation}|{severity}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed)


def apply_degradation(img: Image.Image, path: str, degradation: str, severity: float) -> Image.Image:
    if degradation == "clean":
        return img

    if degradation == "blur":
        if severity <= 0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=float(severity)))

    if degradation == "jpeg":
        if severity <= 0:
            return img
        quality = int(np.clip(100 - severity, 5, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if degradation == "noise":
        if severity <= 0:
            return img
        rng = deterministic_rng(path, degradation, severity)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(severity), arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    if degradation == "downsample":
        factor = max(float(severity), 1.0)
        if factor <= 1:
            return img
        small_w = max(1, int(round(img.width / factor)))
        small_h = max(1, int(round(img.height / factor)))
        return img.resize((small_w, small_h), RESAMPLE_BICUBIC).resize(
            img.size, RESAMPLE_BICUBIC
        )

    raise ValueError(f"Unknown degradation: {degradation}")


def load_frame(path: str, degradation: str, severity: float) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), RESAMPLE_BICUBIC)
    img = apply_degradation(img, path, degradation, severity)
    return tensor_from_pil(img)


class CDFv2DegradedClipDataset(Dataset):
    def __init__(
        self,
        fake_root: Path,
        real_root: Path,
        num_frames: int,
        degradation: str,
        severity: float,
    ):
        self.num_frames = num_frames
        self.degradation = degradation
        self.severity = severity

        vid2paths = defaultdict(list)
        vid2label = {}
        for root, label in [(fake_root, 1), (real_root, 0)]:
            for sample_dir in sorted(root.iterdir()):
                image_path = sample_dir / "image.png"
                if sample_dir.is_dir() and image_path.exists():
                    vid = video_id_from_sample(sample_dir.name)
                    vid2paths[vid].append(str(image_path))
                    vid2label[vid] = label

        self.videos = [
            (vid, sorted(paths), vid2label[vid]) for vid, paths in sorted(vid2paths.items())
        ]
        real_n = sum(1 for _, _, label in self.videos if label == 0)
        fake_n = sum(1 for _, _, label in self.videos if label == 1)
        print(
            f"  CDFv2 clips -> real={real_n} fake={fake_n} total={len(self.videos)} "
            f"for {degradation} severity={severity:g}"
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = sample_frame_indices(len(paths), self.num_frames)
        frames = []
        for frame_idx in indices:
            path = paths[int(frame_idx)]
            try:
                frames.append(load_frame(path, self.degradation, self.severity))
            except Exception:
                frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
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


class TrainRealClipDataset(Dataset):
    def __init__(self, real_train_root: Path, num_frames: int):
        self.num_frames = num_frames
        vid2paths = defaultdict(list)
        for sample_dir in sorted(real_train_root.iterdir()):
            image_path = sample_dir / "image.png"
            if sample_dir.is_dir() and image_path.exists():
                vid2paths[video_id_from_sample(sample_dir.name)].append(str(image_path))
        self.videos = [(vid, sorted(paths)) for vid, paths in sorted(vid2paths.items())]
        print(f"  RVMB source -> {len(self.videos)} clean real training videos")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths = self.videos[idx]
        indices = sample_frame_indices(len(paths), self.num_frames)
        frames = []
        for frame_idx in indices:
            path = paths[int(frame_idx)]
            try:
                frames.append(load_frame(path, "clean", 0.0))
            except Exception:
                frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        return torch.stack(frames, dim=0), 0


def real_clip_collate_fn(batch):
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for frames in frames_list:
        pad_t = max_len - frames.size(0)
        if pad_t > 0:
            frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded.append(frames)
    return torch.stack(padded, dim=0), torch.tensor(labels, dtype=torch.long), lengths


def build_memory_bank_for_inference(
    model: VideoViT,
    real_train_root: Path,
    num_frames: int,
    k: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> RealVideoMemoryBank:
    print("\nBuilding clean real-video memory bank ...")
    dataset = TrainRealClipDataset(real_train_root, num_frames=num_frames)
    if len(dataset) == 0:
        raise RuntimeError(f"No real training videos found in {real_train_root}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=real_clip_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    bank = RealVideoMemoryBank(
        embed_dim=VideoViT.EMBED_DIM,
        num_heads=VideoViT.NUM_TEMPORAL_HEADS,
        k=k,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, _, lengths in tqdm(loader, desc="Building RVMB", leave=False):
            bsz, time_steps, channels, height, width = frames.shape
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            flat_frames = frames.reshape(bsz * time_steps, channels, height, width)
            _, _, cls_list = raw_model.frame_model(flat_frames)

            time_idx = torch.arange(time_steps, device=device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.unsqueeze(1)
            cls_sequences = [
                cls_tokens.reshape(bsz, time_steps, raw_model.EMBED_DIM)
                for cls_tokens in cls_list
            ]
            bank.add(cls_sequences, key_padding_mask)

    bank.build()
    print(f"Memory bank ready: {len(bank)} real-video prototypes per stream, k={k}")
    return bank


def load_model(checkpoint_path: str, num_frames: int, device: torch.device) -> VideoViT:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    fusion_key = "fusion_classifier.weight"
    if fusion_key not in ckpt:
        raise KeyError(f"Cannot find {fusion_key} in checkpoint.")

    expected_dim = VideoViT.NUM_TEMPORAL_HEADS * VideoViT.EMBED_DIM + 2
    actual_dim = int(ckpt[fusion_key].shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Unexpected fusion classifier input dim {actual_dim}; "
            f"expected {expected_dim} for frame-end Stage 2."
        )

    if "memory_gate" not in ckpt:
        raise ValueError(
            "This checkpoint has no memory_gate. Use the original Stage-2 RVMB checkpoint."
        )

    model = VideoViT(num_frames=num_frames, use_memory_bank=True).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("Loaded original Stage-2 RVMB checkpoint.")
    return model


def compute_metrics(labels: list[int], probs: list[float]) -> dict[str, float]:
    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs)
    preds = (probs_np >= 0.5).astype(int)

    fpr_arr, tpr_arr, _ = roc_curve(labels_np, probs_np, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr)))
    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds).ravel()
    return {
        "auc": float(roc_auc_score(labels_np, probs_np)),
        "ap": float(average_precision_score(labels_np, probs_np)),
        "acc": float(accuracy_score(labels_np, preds)),
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "eer": eer,
        "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_degradation(
    model,
    fake_root: Path,
    real_root: Path,
    degradation: str,
    severity: float,
    args,
    device: torch.device,
):
    dataset = CDFv2DegradedClipDataset(
        fake_root=fake_root,
        real_root=real_root,
        num_frames=args.num_frames,
        degradation=degradation,
        severity=severity,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not args.fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    labels, probs, probs_no_frame, vids = [], [], [], []
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        desc = f"{degradation}:{severity:g}"
        for frames, batch_labels, lengths, batch_vids in tqdm(loader, desc=desc, unit="batch"):
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits, logits_no_frame, _, _, _ = model(frames, lengths)
            probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist())
            probs_no_frame.extend(
                torch.softmax(logits_no_frame.float(), dim=1)[:, 1].cpu().numpy().tolist()
            )
            labels.extend(batch_labels.numpy().tolist())
            vids.extend(batch_vids)

    metrics = compute_metrics(labels, probs)
    metrics_no_frame = compute_metrics(labels, probs_no_frame)
    print(
        f"  {degradation:10s} severity={severity:g} | "
        f"AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} "
        f"ACC={metrics['acc'] * 100:.2f}% EER={metrics['eer'] * 100:.2f}% "
        f"no-frame AUC={metrics_no_frame['auc']:.4f}"
    )

    per_video = [
        {
            "video_id": vid,
            "label": label,
            "prob_fake": prob,
            "prob_fake_no_frame": prob_no_frame,
            "degradation": degradation,
            "severity": severity,
        }
        for vid, label, prob, prob_no_frame in zip(vids, labels, probs, probs_no_frame)
    ]
    return metrics, metrics_no_frame, per_video, len(dataset)


def degradation_levels(args) -> dict[str, list[float]]:
    requested = [x.strip().lower() for x in args.degradations.split(",") if x.strip()]
    level_map = {
        "blur": parse_float_list(args.blur_levels),
        "jpeg": parse_float_list(args.jpeg_levels),
        "noise": parse_float_list(args.noise_levels),
        "downsample": parse_float_list(args.downsample_levels),
    }
    unknown = sorted(set(requested) - set(level_map))
    if unknown:
        raise ValueError(f"Unknown degradations: {unknown}")
    return {name: level_map[name] for name in requested}


def write_summary_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "degradation",
        "severity",
        "n_videos",
        "auc",
        "ap",
        "acc",
        "f1",
        "eer",
        "tpr",
        "fpr",
        "auc_no_frame",
        "ap_no_frame",
        "acc_no_frame",
        "f1_no_frame",
        "eer_no_frame",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_results(rows: list[dict], out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARNING] Could not import matplotlib; skipping plot: {exc}")
        return

    by_deg = defaultdict(list)
    for row in rows:
        by_deg[row["degradation"]].append(row)

    n = len(by_deg)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    colors = {
        "blur": "#0057FF",
        "jpeg": "#E31A1C",
        "noise": "#111111",
        "downsample": "#FFB000",
    }

    for ax, (degradation, deg_rows) in zip(axes[0], by_deg.items()):
        deg_rows = sorted(deg_rows, key=lambda r: float(r["severity"]))
        x = [float(r["severity"]) for r in deg_rows]
        y = [float(r["auc"]) * 100.0 for r in deg_rows]
        y_no_frame = [float(r["auc_no_frame"]) * 100.0 for r in deg_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=colors.get(degradation, "#0057FF"),
            label="final",
        )
        ax.plot(
            x,
            y_no_frame,
            marker="s",
            linewidth=1.5,
            markersize=3.5,
            linestyle="--",
            color="#777777",
            label="no frame logits",
        )
        ax.set_title(degradation)
        ax.set_xlabel("degradation strength")
        ax.set_ylabel("Video AUC (%)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(max(0, min(y + y_no_frame) - 5), 100.5)
        if degradation == "jpeg":
            ax.text(
                0.02,
                0.03,
                "quality = 100 - strength",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
            )
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("CDFv2 Stage-2 RVMB Robustness Under Input Degradation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_root = Path(args.fake_root)
    real_root = Path(args.real_root)
    train_real_root = Path(args.train_real_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("=" * 88)
    print("CDFv2 Stage-2 RVMB input degradation evaluation")
    print("=" * 88)
    print(f"Device          : {device}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Fake root       : {fake_root}")
    print(f"Real root       : {real_root}")
    print(f"Train real root : {train_real_root}")
    print(f"Output dir      : {out_dir}")
    print(f"Num frames      : {args.num_frames}")
    print(f"Precision       : {'FP32' if args.fp32 else 'FP16 autocast'}")
    print(f"Degradations    : {args.degradations}")

    model = load_model(args.checkpoint, args.num_frames, device)
    bank = build_memory_bank_for_inference(
        model=model,
        real_train_root=train_real_root,
        num_frames=args.num_frames,
        k=args.knn_k,
        batch_size=args.bank_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    model.attach_memory_bank(bank)

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    rows = []
    levels = degradation_levels(args)
    for degradation, severities in levels.items():
        print("\n" + "=" * 88)
        print(f"Evaluating degradation: {degradation}")
        print("=" * 88)
        for severity in severities:
            metrics, metrics_no_frame, per_video, n_videos = evaluate_degradation(
                model=model,
                fake_root=fake_root,
                real_root=real_root,
                degradation=degradation,
                severity=severity,
                args=args,
                device=device,
            )
            rows.append({
                "degradation": degradation,
                "severity": severity,
                "n_videos": n_videos,
                **metrics,
                "auc_no_frame": metrics_no_frame["auc"],
                "ap_no_frame": metrics_no_frame["ap"],
                "acc_no_frame": metrics_no_frame["acc"],
                "f1_no_frame": metrics_no_frame["f1"],
                "eer_no_frame": metrics_no_frame["eer"],
            })
            if args.save_per_video:
                per_video_path = out_dir / f"per_video_{degradation}_{severity:g}.csv"
                with open(per_video_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_video[0].keys()))
                    writer.writeheader()
                    writer.writerows(per_video)

    summary_path = out_dir / "cdfv2_stage2_degradation_metrics.csv"
    plot_path = out_dir / "cdfv2_stage2_degradation_auc.png"
    write_summary_csv(summary_path, rows)
    plot_results(rows, plot_path)

    print("\n" + "=" * 88)
    print("Final degradation summary")
    print("=" * 88)
    for row in rows:
        print(
            f"{row['degradation']:10s} severity={row['severity']:>6g} "
            f"AUC={row['auc']:.4f} AP={row['ap']:.4f} "
            f"ACC={row['acc'] * 100:.2f}% no-frame AUC={row['auc_no_frame']:.4f}"
        )
    print(f"\nSaved CSV : {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
