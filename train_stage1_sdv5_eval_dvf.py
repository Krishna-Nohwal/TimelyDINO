"""
Train Stage 1 on SDV5 image folders and evaluate DVF every epoch.

Training data:
    imagenet_ai_0424_sdv5/train/
      nature/   -> real label 0
      ai/       -> fake label 1

DVF evaluation:
    Uses the extracted-frame manifest produced by extract_dvf_tiny_frames.py.
    Reports both frame-level AUC and video-level AUC, where video probability
    is the mean fake probability over all extracted frames from that video.

Example:
    python train_stage1_sdv5_eval_dvf.py \
        --sdv5_root /media/tarun/B482367C823642E2/usr/gen/imagenet_ai_0424_sdv5/train \
        --dvf_root /media/tarun/B482367C823642E2/usr/dvf/DVF_tiny_16f \
        --dvf_manifest /media/tarun/B482367C823642E2/usr/dvf/DVF_tiny_16f/manifest_dvf_tiny_16f.csv \
        --save_root checkpoints_stage1_sdv5_dvf
"""

import argparse
import hashlib
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image, ImageOps
from pytorch_metric_learning.losses import MultiSimilarityLoss, SupConLoss
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

from frame_model import ViT


IMG_SIZE = 256
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: train on SDV5 folders, evaluate frame/video AUC on DVF."
    )
    parser.add_argument("--sdv5_root", required=True, type=str)
    parser.add_argument("--test_dataset", default="dvf", choices=["dvf", "genvideo"])
    parser.add_argument("--dvf_root", default="", type=str)
    parser.add_argument("--dvf_manifest", default="", type=str)
    parser.add_argument(
        "--dvf_real_videos",
        default=0,
        type=int,
        help="Number of real DVF videos to evaluate. 0 means use all real videos.",
    )
    parser.add_argument(
        "--dvf_fake_videos",
        default=0,
        type=int,
        help="Number of fake DVF videos to evaluate. 0 means use all fake videos.",
    )
    parser.add_argument(
        "--dvf_sample_seed",
        default=42,
        type=int,
        help="Seed used when subsampling DVF real/fake videos.",
    )
    parser.add_argument(
        "--genvideo_root",
        default="",
        type=str,
        help="Path to GenVideo root containing OpenSora and ZeroScope.",
    )
    parser.add_argument("--genvideo_num_frames", default=16, type=int)
    parser.add_argument("--save_root", default="checkpoints_stage1_sdv5_dvf", type=str)
    parser.add_argument("--load_from", default="", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--test_batch_size",
        default=8,
        type=int,
        help="Evaluation batch size. For GenVideo this is videos per batch.",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--val_ratio", default=0.05, type=float)
    parser.add_argument(
        "--train_real_frames",
        default=0,
        type=int,
        help="Number of real/nature SDV5 training frames to use after val split. 0 means all.",
    )
    parser.add_argument(
        "--train_fake_frames",
        default=0,
        type=int,
        help="Number of fake/ai SDV5 training frames to use after val split. 0 means all.",
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=512, type=int)
    parser.add_argument("--supcon_weight", default=1 / 16, type=float)
    parser.add_argument("--ms_weight", default=1 / 16, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--hash_leak_check",
        action="store_true",
        help="Also compare 256x256 RGB image hashes between SDV5 train/val and DVF.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj)))
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def link_or_copy_checkpoint(src: Path, dst: Path) -> str:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        print(f"  Could not hard-link best checkpoint ({exc}); falling back to copy.")
        shutil.copy2(src, dst)
        return "copy"


def load_image(path: str, train: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    if train and random.random() < 0.5:
        image = ImageOps.mirror(image)
    return image_to_tensor(image)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def image_content_hash(path: str) -> str:
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def audit_dvf_leakage(train_entries, val_entries, dvf_dataset, hash_check: bool):
    train_paths = {str(Path(path).expanduser().resolve()) for path, _ in train_entries}
    val_paths = {str(Path(path).expanduser().resolve()) for path, _ in val_entries}
    dvf_paths = {str(Path(row[0]).expanduser().resolve()) for row in dvf_dataset.rows}

    train_overlap = train_paths & dvf_paths
    val_overlap = val_paths & dvf_paths

    print("\nLeakage audit: SDV5 vs DVF")
    print(f"  SDV5 train images: {len(train_paths)}")
    print(f"  SDV5 val images:   {len(val_paths)}")
    print(f"  DVF eval frames:   {len(dvf_paths)}")
    print(f"  exact path overlap train vs DVF: {len(train_overlap)}")
    print(f"  exact path overlap val   vs DVF: {len(val_overlap)}")

    if train_overlap or val_overlap:
        examples = sorted((train_overlap | val_overlap))[:10]
        for path in examples:
            print(f"    overlap: {path}")
        raise RuntimeError("DVF leakage detected: exact file path overlap with SDV5.")

    if not hash_check:
        print("  image hash overlap: skipped (pass --hash_leak_check to enable)")
        print("Leakage audit passed: no exact path overlap.\n")
        return

    print("  Computing 256x256 RGB image hashes for content-overlap check ...")
    train_hashes = defaultdict(list)
    val_hashes = defaultdict(list)
    dvf_hashes = defaultdict(list)

    for path, _ in tqdm(train_entries, desc="Hash SDV5 train", leave=False):
        train_hashes[image_content_hash(path)].append(path)
    for path, _ in tqdm(val_entries, desc="Hash SDV5 val", leave=False):
        val_hashes[image_content_hash(path)].append(path)
    for image_path, _, _, _, _ in tqdm(dvf_dataset.rows, desc="Hash DVF", leave=False):
        dvf_hashes[image_content_hash(image_path)].append(image_path)

    train_hash_overlap = set(train_hashes) & set(dvf_hashes)
    val_hash_overlap = set(val_hashes) & set(dvf_hashes)
    print(f"  image hash overlap train vs DVF: {len(train_hash_overlap)}")
    print(f"  image hash overlap val   vs DVF: {len(val_hash_overlap)}")

    if train_hash_overlap or val_hash_overlap:
        for digest in sorted(train_hash_overlap | val_hash_overlap)[:5]:
            print(f"    hash overlap: {digest}")
            for path in train_hashes.get(digest, [])[:3]:
                print(f"      train: {path}")
            for path in val_hashes.get(digest, [])[:3]:
                print(f"      val:   {path}")
            for path in dvf_hashes.get(digest, [])[:3]:
                print(f"      dvf:   {path}")
        raise RuntimeError("DVF leakage detected: image-content hash overlap with SDV5.")

    print("Leakage audit passed: no exact path or image-content overlap.\n")


def discover_sdv5_images(root: str):
    root_path = Path(root).expanduser()
    ai_dir = root_path / "ai"
    nature_dir = root_path / "nature"
    if not ai_dir.is_dir():
        raise FileNotFoundError(f"Missing fake folder: {ai_dir}")
    if not nature_dir.is_dir():
        raise FileNotFoundError(f"Missing real folder: {nature_dir}")

    def collect(folder: Path, label: int):
        paths = [
            str(path)
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        return [(path, label) for path in sorted(paths)]

    real_entries = collect(nature_dir, 0)
    fake_entries = collect(ai_dir, 1)
    print(f"SDV5 discovered -> Real/nature: {len(real_entries)} | Fake/ai: {len(fake_entries)}")
    if not real_entries or not fake_entries:
        raise RuntimeError("Both SDV5 classes must contain at least one image.")
    return real_entries, fake_entries


def split_entries(
    real_entries,
    fake_entries,
    val_ratio: float,
    seed: int,
    train_real_frames: int = 0,
    train_fake_frames: int = 0,
):
    rng = np.random.default_rng(seed)
    real = list(real_entries)
    fake = list(fake_entries)
    rng.shuffle(real)
    rng.shuffle(fake)

    real_val_n = max(1, int(len(real) * val_ratio)) if len(real) > 1 else 0
    fake_val_n = max(1, int(len(fake) * val_ratio)) if len(fake) > 1 else 0

    val_entries = real[:real_val_n] + fake[:fake_val_n]
    real_train = real[real_val_n:]
    fake_train = fake[fake_val_n:]

    if train_real_frames > 0:
        if train_real_frames > len(real_train):
            print(
                f"Requested {train_real_frames} real train frames, "
                f"but only {len(real_train)} available; using all."
            )
        else:
            real_train = real_train[:train_real_frames]

    if train_fake_frames > 0:
        if train_fake_frames > len(fake_train):
            print(
                f"Requested {train_fake_frames} fake train frames, "
                f"but only {len(fake_train)} available; using all."
            )
        else:
            fake_train = fake_train[:train_fake_frames]

    train_entries = real_train + fake_train
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)

    print(
        f"SDV5 train -> Real: {sum(y == 0 for _, y in train_entries)} | "
        f"Fake: {sum(y == 1 for _, y in train_entries)} | Total: {len(train_entries)}"
    )
    print(
        f"SDV5 val   -> Real: {sum(y == 0 for _, y in val_entries)} | "
        f"Fake: {sum(y == 1 for _, y in val_entries)} | Total: {len(val_entries)}"
    )
    return train_entries, val_entries


class ImageEntriesDataset(Dataset):
    def __init__(self, entries: list[tuple[str, int]], train: bool):
        self.entries = entries
        self.train = train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        return load_image(path, self.train), int(label)


def extract_video_id(sample_dir: str) -> str:
    sample_dir = str(sample_dir).replace("\\", "/")
    path = Path(sample_dir)
    base = path.name
    prefix = path.parent.as_posix()
    marker = "_frame_"
    if marker in base:
        base = base[:base.rfind(marker)]
    return f"{prefix}/{base}" if prefix not in ("", ".") else base


class DVFFrameDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        root_dir: str,
        real_videos: int = 0,
        fake_videos: int = 0,
        sample_seed: int = 42,
    ):
        df = pd.read_csv(manifest_csv)
        required = {"sample_dir", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"DVF manifest must contain {required}. Found {list(df.columns)}")

        root = Path(root_dir).expanduser()
        rows = []
        skipped = 0
        for _, row in df.iterrows():
            sample_dir = str(row["sample_dir"]).replace("\\", "/")
            label = int(row["label"])
            video_id = str(row["video_id"]) if "video_id" in df.columns else extract_video_id(sample_dir)
            if "source" in df.columns:
                source = str(row["source"])
            else:
                parts = Path(sample_dir).parts
                if parts and parts[0] in {"real", "fake"} and len(parts) > 1:
                    source = parts[1]
                else:
                    source = parts[0] if parts else "unknown"
            image_path = root / sample_dir / "image.png"
            if image_path.is_file():
                rows.append((str(image_path), label, video_id, sample_dir, source))
            else:
                skipped += 1

        rows = self._subsample_videos(rows, real_videos, fake_videos, sample_seed)
        self.rows = rows
        print(
            f"DVF frames -> Real: {sum(r[1] == 0 for r in rows)} | "
            f"Fake: {sum(r[1] == 1 for r in rows)} | Total: {len(rows)}"
        )
        print(f"DVF videos -> {len(set(r[2] for r in rows))}")
        print("DVF subsets:")
        for source in sorted(set(r[4] for r in rows)):
            source_rows = [r for r in rows if r[4] == source]
            source_videos = set(r[2] for r in source_rows)
            print(
                f"  {source}: frames={len(source_rows)} videos={len(source_videos)} "
                f"real_frames={sum(r[1] == 0 for r in source_rows)} "
                f"fake_frames={sum(r[1] == 1 for r in source_rows)}"
            )
        if skipped:
            print(f"  [DVF] skipped missing image.png rows: {skipped}")
        if not rows:
            raise RuntimeError("No DVF frames found.")

    @staticmethod
    def _subsample_videos(rows, real_videos: int, fake_videos: int, sample_seed: int):
        if real_videos <= 0 and fake_videos <= 0:
            return rows

        video_labels = {}
        for row in rows:
            _, label, video_id, _, _ = row
            video_labels[str(video_id)] = int(label)

        real_ids = sorted([vid for vid, label in video_labels.items() if label == 0])
        fake_ids = sorted([vid for vid, label in video_labels.items() if label == 1])
        rng = np.random.default_rng(sample_seed)

        def choose(ids, requested, name):
            if requested <= 0:
                print(f"DVF {name} video sampling: using all {len(ids)} videos")
                return set(ids)
            if requested > len(ids):
                print(
                    f"DVF {name} video sampling: requested {requested}, "
                    f"but only {len(ids)} available; using all"
                )
                return set(ids)
            selected = sorted(rng.choice(ids, size=requested, replace=False).tolist())
            print(f"DVF {name} video sampling: selected {len(selected)} / {len(ids)} videos")
            return set(selected)

        keep_real = choose(real_ids, real_videos, "real")
        keep_fake = choose(fake_ids, fake_videos, "fake")
        keep = keep_real | keep_fake
        filtered = [row for row in rows if str(row[2]) in keep]
        print(
            f"DVF video subsample -> real videos: {len(keep_real)} | "
            f"fake videos: {len(keep_fake)} | frames: {len(filtered)}"
        )
        return filtered

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_path, label, video_id, sample_dir, source = self.rows[idx]
        return load_image(image_path, train=False), int(label), video_id, sample_dir, source


def normalize_source_name(name: str) -> str:
    lowered = name.lower().replace("_", "").replace("-", "")
    if "opensora" in lowered:
        return "opensora"
    if "zeroscope" in lowered:
        return "zeroscope"
    return name.lower()


class GenVideoDataset(Dataset):
    """Raw GenVideo video dataset. All videos are fake (label=1)."""

    def __init__(self, root_dir: str, num_frames: int = 16):
        root = Path(root_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"GenVideo root not found: {root}")

        rows = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            rel = path.relative_to(root)
            source = normalize_source_name(rel.parts[0] if rel.parts else path.parent.name)
            video_id = rel.with_suffix("").as_posix()
            rows.append((str(path), 1, video_id, source))

        self.rows = rows
        self.num_frames = int(num_frames)
        print(f"GenVideo videos -> Fake: {len(rows)} | Real: 0 | Total: {len(rows)}")
        print("GenVideo subsets:")
        for source in sorted(set(row[3] for row in rows)):
            count = sum(row[3] == source for row in rows)
            print(f"  {source}: videos={count} frames_per_video={self.num_frames}")
        if not rows:
            raise RuntimeError(f"No GenVideo videos found under {root}")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _read_frame_at(cap, frame_idx: int):
        import cv2

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _read_all_frames(video_path: str):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, Image.fromarray(frame).convert("RGB")))
            idx += 1
        cap.release()
        return frames

    def _sample_frames(self, video_path: str) -> tuple[torch.Tensor, int]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        images = []
        if total > 0:
            targets = np.linspace(0, max(total - 1, 0), self.num_frames)
            targets = np.round(targets).astype(int).tolist()
            for target in targets:
                image = self._read_frame_at(cap, target)
                if image is None:
                    images = []
                    break
                images.append(image)
        cap.release()

        if len(images) != self.num_frames:
            all_frames = self._read_all_frames(video_path)
            if not all_frames:
                raise RuntimeError(f"No decodable frames in video: {video_path}")
            targets = np.linspace(0, len(all_frames) - 1, self.num_frames)
            targets = np.round(targets).astype(int).tolist()
            images = [all_frames[i][1] for i in targets]
            total = len(all_frames)

        frames = torch.stack([image_to_tensor(image) for image in images], dim=0)
        return frames, int(total)

    def __getitem__(self, idx):
        video_path, label, video_id, source = self.rows[idx]
        frames, total = self._sample_frames(video_path)
        return frames, int(label), video_id, source, video_path, total


def compute_metrics(labels, probs, split_name: str, epoch: int):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int64)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs, pos_label=1)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    print(
        f"  [{split_name}] Epoch {epoch + 1:02d} | "
        f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc * 100:.2f}%  F1={f1:.4f}  "
        f"EER={eer * 100:.2f}%  TPR={tpr * 100:.2f}%  FPR={fpr * 100:.2f}%  "
        f"TNR={tnr * 100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return auc


def compute_metrics_safe(labels, probs, split_name: str, epoch: int):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    real_n = int((labels == 0).sum())
    fake_n = int((labels == 1).sum())

    if len(labels) == 0:
        print(f"  [{split_name}] Epoch {epoch + 1:02d} | EMPTY")
        return float("nan")

    if real_n == 0 or fake_n == 0:
        preds = (probs >= 0.5).astype(np.int64)
        acc = accuracy_score(labels, preds)
        mean_prob = float(probs.mean())
        min_prob = float(probs.min())
        max_prob = float(probs.max())
        print(
            f"  [{split_name}] Epoch {epoch + 1:02d} | "
            f"AUC=nan  Acc={acc * 100:.2f}%  "
            f"real={real_n} fake={fake_n}  "
            f"p_fake mean={mean_prob:.4f} min={min_prob:.4f} max={max_prob:.4f} "
            f"(single-class subset)"
        )
        return float("nan")

    return compute_metrics(labels, probs, split_name, epoch)


bce_loss = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()
ms_loss = MultiSimilarityLoss()


def cls_loss(logits, features, labels, lam_supcon, lam_ms):
    features_norm = torch.nn.functional.normalize(features, dim=1)
    return (
        bce_loss(logits, labels)
        + lam_supcon * supcon_loss(features, labels)
        + lam_ms * ms_loss(features_norm, labels)
    )


def autocast_context(device: torch.device, amp_enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)


@torch.no_grad()
def eval_image_loader(model, loader, desc: str, device: torch.device, amp_enabled: bool):
    model.eval()
    labels_all, probs_all = [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits_list, _, _ = model(imgs)
            probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.tolist())
    return labels_all, probs_all


@torch.no_grad()
def eval_dvf(model, loader, desc: str, device: torch.device, amp_enabled: bool):
    model.eval()
    frame_labels, frame_probs = [], []
    video_probs = defaultdict(list)
    video_labels = {}
    subset_frame_labels = defaultdict(list)
    subset_frame_probs = defaultdict(list)
    subset_video_probs = defaultdict(lambda: defaultdict(list))
    subset_video_labels = defaultdict(dict)

    for imgs, labels, video_ids, _, sources in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits_list, _, _ = model(imgs)
            probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()

        label_list = labels.numpy().astype(int).tolist()
        frame_labels.extend(label_list)
        frame_probs.extend(probs.tolist())
        for video_id, label, prob, source in zip(video_ids, label_list, probs.tolist(), sources):
            source = str(source)
            video_id = str(video_id)
            video_probs[str(video_id)].append(float(prob))
            video_labels[str(video_id)] = int(label)
            subset_frame_labels[source].append(int(label))
            subset_frame_probs[source].append(float(prob))
            subset_video_probs[source][video_id].append(float(prob))
            subset_video_labels[source][video_id] = int(label)

    vid_ids = sorted(video_probs)
    vid_labels = [video_labels[vid] for vid in vid_ids]
    vid_probs = [float(np.mean(video_probs[vid])) for vid in vid_ids]
    subset_metrics = {}
    for source in sorted(subset_frame_labels):
        source_vid_ids = sorted(subset_video_probs[source])
        subset_metrics[source] = {
            "frame_labels": subset_frame_labels[source],
            "frame_probs": subset_frame_probs[source],
            "video_labels": [subset_video_labels[source][vid] for vid in source_vid_ids],
            "video_probs": [float(np.mean(subset_video_probs[source][vid])) for vid in source_vid_ids],
        }
    return frame_labels, frame_probs, vid_labels, vid_probs, subset_metrics


@torch.no_grad()
def eval_genvideo(model, loader, desc: str, device: torch.device, amp_enabled: bool):
    model.eval()
    frame_labels, frame_probs = [], []
    video_labels, video_probs = [], []
    subset_frame_labels = defaultdict(list)
    subset_frame_probs = defaultdict(list)
    subset_video_labels = defaultdict(list)
    subset_video_probs = defaultdict(list)

    for frames, labels, video_ids, sources, _, totals in tqdm(loader, desc=desc, leave=False):
        # frames: (B, T, C, H, W)
        bsz, time_steps, channels, height, width = frames.shape
        flat = frames.reshape(bsz * time_steps, channels, height, width).to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits_list, _, _ = model(flat)
            probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1]
        probs = probs.reshape(bsz, time_steps).cpu().numpy()

        label_list = labels.numpy().astype(int).tolist()
        for i, (label, video_id, source) in enumerate(zip(label_list, video_ids, sources)):
            source = str(source)
            frame_prob_list = probs[i].astype(float).tolist()
            video_prob = float(np.mean(frame_prob_list))

            frame_labels.extend([int(label)] * time_steps)
            frame_probs.extend(frame_prob_list)
            video_labels.append(int(label))
            video_probs.append(video_prob)

            subset_frame_labels[source].extend([int(label)] * time_steps)
            subset_frame_probs[source].extend(frame_prob_list)
            subset_video_labels[source].append(int(label))
            subset_video_probs[source].append(video_prob)

    subset_metrics = {}
    for source in sorted(subset_video_labels):
        subset_metrics[source] = {
            "frame_labels": subset_frame_labels[source],
            "frame_probs": subset_frame_probs[source],
            "video_labels": subset_video_labels[source],
            "video_probs": subset_video_probs[source],
        }
    return frame_labels, frame_probs, video_labels, video_probs, subset_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and not args.no_amp
    torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print("Labels: nature=real(0), ai=fake(1)")

    real_entries, fake_entries = discover_sdv5_images(args.sdv5_root)
    train_entries, val_entries = split_entries(
        real_entries,
        fake_entries,
        args.val_ratio,
        args.seed,
        train_real_frames=args.train_real_frames,
        train_fake_frames=args.train_fake_frames,
    )

    train_dataset = ImageEntriesDataset(train_entries, train=True)
    val_dataset = ImageEntriesDataset(val_entries, train=False)
    if args.test_dataset == "dvf":
        if not args.dvf_root or not args.dvf_manifest:
            raise ValueError("--dvf_root and --dvf_manifest are required when --test_dataset dvf")
        test_dataset = DVFFrameDataset(
            args.dvf_manifest,
            args.dvf_root,
            real_videos=args.dvf_real_videos,
            fake_videos=args.dvf_fake_videos,
            sample_seed=args.dvf_sample_seed,
        )
        audit_dvf_leakage(train_entries, val_entries, test_dataset, args.hash_leak_check)
    else:
        if not args.genvideo_root:
            raise ValueError("--genvideo_root is required when --test_dataset genvideo")
        if args.hash_leak_check:
            print("--hash_leak_check is only implemented for extracted-frame DVF; skipping for GenVideo.")
        test_dataset = GenVideoDataset(args.genvideo_root, num_frames=args.genvideo_num_frames)

    persistent = args.num_workers > 0
    prefetch = 4 if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    model = ViT().to(device)
    if args.load_from:
        state = clean_state_dict(torch.load(args.load_from, map_location="cpu"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.load_from}")
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    iter_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * iter_per_epoch)
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
    lr_min = 1e-6 / args.lr

    def lr_lambda(step):
        if step <= warmup_steps:
            return step / warmup_steps + lr_min
        denom = max(1, total_steps - warmup_steps)
        return ((1 + math.cos((step - warmup_steps) * math.pi / denom)) / 2) + lr_min

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_test_score = -1.0
    best_epoch = -1
    sep = "=" * 88

    for epoch in range(args.epochs):
        print(f"\n{sep}")
        print(f"  EPOCH {epoch + 1}/{args.epochs}")
        print(sep)

        model.train()
        train_labels, train_probs = [], []
        running_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1} [SDV5 train]", leave=False)
        ):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device, amp_enabled):
                logits_list, features_list, _ = model(imgs)
                l_primary = cls_loss(
                    logits_list[3], features_list[3], labels,
                    args.supcon_weight, args.ms_weight,
                )
                l_aux = (
                    cls_loss(logits_list[0], features_list[0], labels, args.supcon_weight, args.ms_weight)
                    + cls_loss(logits_list[1], features_list[1], labels, args.supcon_weight, args.ms_weight)
                    + cls_loss(logits_list[2], features_list[2], labels, args.supcon_weight, args.ms_weight)
                ) / 3.0
                loss = l_primary + l_aux

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            with torch.no_grad():
                probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].detach().cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())

            if batch_idx % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  batch={batch_idx:5d}/{iter_per_epoch}  "
                    f"loss={loss.item():.4f}  lr={lr_now:.3e}"
                )

        mean_loss = running_loss / max(1, iter_per_epoch)
        print(f"\n  mean train loss: {mean_loss:.4f}")
        compute_metrics(train_labels, train_probs, "SDV5 Train frame", epoch)

        val_labels, val_probs = eval_image_loader(
            model, val_loader, f"Epoch {epoch + 1} [SDV5 val]", device, amp_enabled
        )
        compute_metrics(val_labels, val_probs, "SDV5 Val frame  ", epoch)

        if args.test_dataset == "dvf":
            test_fl, test_fp, test_vl, test_vp, test_subsets = eval_dvf(
                model, test_loader, f"Epoch {epoch + 1} [DVF]", device, amp_enabled
            )
            frame_metric = compute_metrics(test_fl, test_fp, "DVF Frame      ", epoch)
            video_metric = compute_metrics(test_vl, test_vp, "DVF Video-mean ", epoch)
            print(
                f"  [DVF summary] frame_auc={frame_metric:.4f}  "
                f"video_mean_auc={video_metric:.4f}"
            )
            print("  [DVF subsets]")
            for source, data in test_subsets.items():
                frame_auc = compute_metrics_safe(
                    data["frame_labels"], data["frame_probs"],
                    f"DVF/{source} Frame", epoch,
                )
                video_auc = compute_metrics_safe(
                    data["video_labels"], data["video_probs"],
                    f"DVF/{source} Video", epoch,
                )
                print(
                    f"    {source}: frame_auc={frame_auc:.4f}  "
                    f"video_mean_auc={video_auc:.4f}  "
                    f"frames={len(data['frame_labels'])} videos={len(data['video_labels'])}"
                )
            test_score = video_metric
            best_metric_name = "DVF video-mean AUC"
        else:
            test_fl, test_fp, test_vl, test_vp, test_subsets = eval_genvideo(
                model, test_loader, f"Epoch {epoch + 1} [GenVideo]", device, amp_enabled
            )
            compute_metrics_safe(test_fl, test_fp, "GenVideo Frame ", epoch)
            compute_metrics_safe(test_vl, test_vp, "GenVideo Video ", epoch)
            video_acc = float(((np.asarray(test_vp) >= 0.5).astype(np.int64) == np.asarray(test_vl)).mean())
            video_mean = float(np.mean(test_vp))
            print(
                f"  [GenVideo summary] video_fake_acc={video_acc:.4f}  "
                f"video_mean_p_fake={video_mean:.4f}"
            )
            print("  [GenVideo subsets]")
            for source, data in test_subsets.items():
                compute_metrics_safe(
                    data["frame_labels"], data["frame_probs"],
                    f"GenVideo/{source} Frame", epoch,
                )
                compute_metrics_safe(
                    data["video_labels"], data["video_probs"],
                    f"GenVideo/{source} Video", epoch,
                )
                source_video_acc = float(
                    ((np.asarray(data["video_probs"]) >= 0.5).astype(np.int64)
                     == np.asarray(data["video_labels"])).mean()
                )
                print(
                    f"    {source}: video_fake_acc={source_video_acc:.4f}  "
                    f"video_mean_p_fake={float(np.mean(data['video_probs'])):.4f}  "
                    f"frames={len(data['frame_labels'])} videos={len(data['video_labels'])}"
                )
            test_score = video_acc
            best_metric_name = "GenVideo video fake accuracy"

        live_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = live_model.state_dict()
        latest_path = Path(args.save_root) / "latest.pth"
        best_path = Path(args.save_root) / "best.pth"
        atomic_torch_save(state_dict, latest_path)
        live_model.vit.save_pretrained(Path(args.save_root) / "latest_lora")

        if test_score > best_test_score:
            best_test_score = test_score
            best_epoch = epoch
            link_mode = link_or_copy_checkpoint(latest_path, best_path)
            live_model.vit.save_pretrained(Path(args.save_root) / "best_lora")
            print(
                f"\n  * New best {best_metric_name}={best_test_score:.4f} "
                f"-> saved best.pth ({link_mode})"
            )
        else:
            print(f"\n  Best so far: epoch {best_epoch + 1}  {best_metric_name}={best_test_score:.4f}")

    print(f"\n{sep}")
    print(
        f"  Training complete. Best checkpoint: epoch {best_epoch + 1}  "
        f"{best_metric_name}={best_test_score:.4f}"
    )
    print(f"  Saved to: {Path(args.save_root) / 'best.pth'}")
    print(sep)


if __name__ == "__main__":
    main()
