"""
UMAP for Stage-1 frame-level embeddings across evaluation datasets.

This script runs frame_model.ViT directly, extracts one embedding per sampled
frame, and plots the resulting frame-level representation space. It uses the
same dataset layouts as tsne_predictions1.py, but does not import that script
so it stays standalone.

Example
-------
python umap_stage1_frame_embeddings.py \
    --checkpoint /home/tarun/Desktop/best/stage1_best.pth \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/ \
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --cdfv3_root /media/tarun/B482367C823642E2/usr/cdfv3_face_crops \
    --cdfv3_csv /media/tarun/B482367C823642E2/usr/cdfv3_face_crops/manifest_cdfv3_face_crops.csv \
    --df0_fake_root /media/tarun/B482367C823642E2/usr/df1.0_faces/fake \
    --df0_real_root /media/tarun/B482367C823642E2/usr/df1.0_faces/real \
    --dfd_fake_root /media/tarun/B482367C823642E2/usr/dfd_faces/fake \
    --dfd_real_root /media/tarun/B482367C823642E2/usr/dfd_faces/real \
    --dfdc_fake_root /media/tarun/B482367C823642E2/usr/dfdc/fake \
    --dfdc_real_root /media/tarun/B482367C823642E2/usr/dfdc/real \
    --wdf_fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --wdf_real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --uadfv_fake_root /media/tarun/B482367C823642E2/usr/uadfv_faces/fake \
    --uadfv_real_root /media/tarun/B482367C823642E2/usr/uadfv_faces/real \
    --frames_per_video 4 \
    --max_frames_per_dataset 0 \
    --feature_source last_features \
    --out umap_stage1_frames.png

To reuse cached frame embeddings:

python umap_stage1_frame_embeddings.py \
    --embeddings_npz umap_stage1_frames_embeddings.npz \
    --out umap_stage1_frames.png
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from frame_model import ViT


IMG_SIZE = 256
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(
        description="UMAP of Stage-1 frame_model.ViT frame-level embeddings."
    )
    p.add_argument("--checkpoint", default="", help="Stage-1 checkpoint path or directory containing best.pth.")
    p.add_argument("--embeddings_npz", default="", help="Cached embeddings produced by this script.")
    p.add_argument("--embeddings_out", default="", help="Where to cache embeddings. Default: next to --out.")
    p.add_argument("--out", default="umap_stage1_frames.png")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=IMG_SIZE)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--feature_source",
        default="last_features",
        choices=["last_features", "concat_features", "last_cls", "concat_cls", "last_logits"],
        help="Stage-1 representation to UMAP. Default is layer-23 SpatialHead feature.",
    )
    p.add_argument("--frames_per_video", type=int, default=4, help="0 = use all frames from selected videos.")
    p.add_argument("--max_videos_per_dataset", type=int, default=0, help="0 = no video cap.")
    p.add_argument("--max_frames_per_dataset", type=int, default=0, help="0 = no frame cap after video/frame sampling.")
    p.add_argument("--val_ratio", type=float, default=0.0, help="FF++ val split ratio. 0 = all FF++ videos.")

    # FF++
    p.add_argument("--manifest", default="")
    p.add_argument("--root_dir", default="")

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # CDFv3 / CDF++
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # DFo / DeeperForensics-1.0 nested roots
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")

    # DFD nested roots
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC flat roots
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")

    # WDF flat roots
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV nested roots
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")

    # UMAP / cleanup
    p.add_argument("--outlier_std", type=float, default=2.0, help="<=0 disables outlier removal.")
    p.add_argument("--outlier_group", default="class", choices=["class", "dataset_class", "dataset", "all"])
    p.add_argument("--umap_neighbors", type=int, default=50)
    p.add_argument("--umap_min_dist", type=float, default=0.02)
    p.add_argument("--umap_metric", default="cosine")
    p.add_argument("--umap_seed", type=int, default=42)
    return p.parse_args()


def default_embeddings_path(out_path: str) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_embeddings.npz"))


def sample_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_target <= 0 or n_available <= n_target:
        return np.arange(n_available)
    return np.linspace(0, n_available - 1, n_target, dtype=int)


def stratified_cap(videos: List[Tuple[str, List[str], int]], cap: int, seed: int):
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


def cap_frames(items: list[dict], max_frames: int, seed: int) -> list[dict]:
    if max_frames <= 0 or len(items) <= max_frames:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=max_frames, replace=False)
    return [items[int(i)] for i in sorted(idx)]


def print_dataset_counts(title: str, labels: np.ndarray, dataset_tags: List[str]):
    tags = np.asarray(dataset_tags)
    print(f"\n{title}")
    print("  dataset      total   real   fake")
    print("  -------------------------------")
    for dset in sorted(set(tags.tolist())):
        mask = tags == dset
        real_n = int(((labels == 0) & mask).sum())
        fake_n = int(((labels == 1) & mask).sum())
        print(f"  {dset:<10} {int(mask.sum()):>5} {real_n:>6} {fake_n:>6}")
    print(f"  {'TOTAL':<10} {len(labels):>5} {int((labels == 0).sum()):>6} {int((labels == 1).sum()):>6}")


def extract_video_id_ffpp(sample_dir: str) -> str:
    parts = Path(str(sample_dir).replace("\\", "/")).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = basename.rfind("_frame_")
    if idx != -1:
        clip_id = basename[:idx]
    else:
        m = re.search(r"_f\d+$", basename)
        clip_id = basename[:m.start()] if m else basename
    return f"{prefix}/{clip_id}" if prefix else clip_id


def build_ffpp_videos(manifest_csv: str, root_dir: str, val_ratio: float) -> List[Tuple[str, List[str], int]]:
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")
    df["video_id"] = df["sample_dir"].apply(extract_video_id_ffpp)

    if val_ratio > 0:
        rng = np.random.default_rng(42)
        real_vids = rng.permutation(df[df["label"] == 0]["video_id"].unique())
        fake_vids = rng.permutation(df[df["label"] == 1]["video_id"].unique())
        real_val_n = max(1, int(len(real_vids) * val_ratio))
        fake_val_n = max(1, int(len(fake_vids) * val_ratio))
        selected_ids = set(real_vids[:real_val_n]) | set(fake_vids[:fake_val_n])
        df = df[df["video_id"].isin(selected_ids)].reset_index(drop=True)
        split_name = f"val split ({val_ratio:.3f})"
    else:
        split_name = "all"

    root = Path(root_dir)
    videos = []
    for video_id, group in df.groupby("video_id"):
        label = int(group["label"].iloc[0])
        paths = []
        for rel in group["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
            path = root / rel / "image.png"
            if path.is_file():
                paths.append(str(path))
        if paths:
            videos.append((str(video_id), sorted(paths), label))
    print_video_counts("FFPP", videos, split_name)
    return videos


def video_id_from_sample(sample_name: str) -> str:
    return re.sub(r"_(?:frame_|f)\d+$", "", sample_name)


def build_cdfv2_videos(fake_root: str, real_root: str) -> List[Tuple[str, List[str], int]]:
    vid2paths, vid2label = defaultdict(list), {}
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [CDFv2] WARNING: missing root {root}")
            continue
        for d in sorted(root.iterdir()):
            path = d / "image.png"
            if d.is_dir() and path.exists():
                vid = video_id_from_sample(d.name)
                vid2paths[vid].append(str(path))
                vid2label[vid] = label
    videos = [(vid, sorted(paths), vid2label[vid]) for vid, paths in sorted(vid2paths.items())]
    print_video_counts("CDFv2", videos)
    return videos


def video_id_from_cdfv3_sample_dir(sample_dir: str) -> str:
    return Path(str(sample_dir).replace("\\", "/")).parent.name


def build_cdfv3_videos(cdfv3_csv: str, cdfv3_root: str) -> List[Tuple[str, List[str], int]]:
    df = pd.read_csv(cdfv3_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CDFv3 manifest must contain {required}. Found: {list(df.columns)}")
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(video_id_from_cdfv3_sample_dir)
    root = Path(cdfv3_root)

    videos = []
    for video_id, group in df.groupby("video_id"):
        manifest_label = int(group["label"].iloc[0])
        label = 0 if manifest_label == 1 else 1
        paths = []
        for rel in group["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
            path = root / rel / "image.png"
            if path.is_file():
                paths.append(str(path))
        if paths:
            videos.append((str(video_id), sorted(paths), label))
    print_video_counts("CDFv3", videos)
    return videos


def sort_nested_frame_paths(paths: List[Path]) -> List[str]:
    def key_fn(path: Path):
        parent = path.parent.name
        if parent.isdigit():
            return int(parent), str(path)
        return 10**12, str(path)
    return [str(path) for path in sorted(paths, key=key_fn)]


def build_nested_image_videos(fake_root: str, real_root: str, dataset_name: str) -> List[Tuple[str, List[str], int]]:
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        for video_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            paths = sort_nested_frame_paths(list(video_dir.rglob("image.png")))
            if paths:
                videos.append((video_dir.name, paths, label))
    print_video_counts(dataset_name, videos)
    return videos


FLAT_FRAME_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def build_flat_videos(fake_root: str, real_root: str, dataset_name: str) -> List[Tuple[str, List[str], int]]:
    videos = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            print(f"  [{dataset_name}] WARNING: missing root {root}")
            continue
        grouped = defaultdict(list)
        skipped = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = FLAT_FRAME_RE.match(path.name)
            if not match:
                skipped += 1
                continue
            video_id, frame_idx = match.group(1), int(match.group(2))
            grouped[video_id].append((frame_idx, str(path)))
        if skipped:
            print(f"  [{dataset_name}] skipped {skipped} files under {root}")
        for video_id, indexed_paths in sorted(grouped.items()):
            paths = [p for _, p in sorted(indexed_paths)]
            if paths:
                videos.append((video_id, paths, label))
    print_video_counts(dataset_name, videos)
    return videos


def print_video_counts(name: str, videos: List[Tuple[str, List[str], int]], extra: str = ""):
    real_n = sum(1 for _, _, label in videos if label == 0)
    fake_n = sum(1 for _, _, label in videos if label == 1)
    suffix = f" [{extra}]" if extra else ""
    print(f"  [{name}{suffix}] {len(videos)} videos  (real={real_n}, fake={fake_n})")


def build_dataset_videos(args) -> dict[str, List[Tuple[str, List[str], int]]]:
    dataset_videos = {}
    if args.manifest and args.root_dir:
        print("\nBuilding FF++ video list ...")
        dataset_videos["FFPP"] = build_ffpp_videos(args.manifest, args.root_dir, args.val_ratio)
    else:
        print("\n[skip] FF++: --manifest / --root_dir not provided.")

    if args.cdfv2_fake_root and args.cdfv2_real_root:
        print("\nBuilding CDFv2 video list ...")
        dataset_videos["CDFv2"] = build_cdfv2_videos(args.cdfv2_fake_root, args.cdfv2_real_root)
    else:
        print("\n[skip] CDFv2: --cdfv2_fake_root / --cdfv2_real_root not provided.")

    if args.cdfv3_root:
        cdfv3_csv = args.cdfv3_csv or str(Path(args.cdfv3_root) / "manifest_cdfv3_face_crops.csv")
        print("\nBuilding CDFv3 video list ...")
        dataset_videos["CDFv3"] = build_cdfv3_videos(cdfv3_csv, args.cdfv3_root)
    else:
        print("\n[skip] CDFv3: --cdfv3_root not provided.")

    dfo_fake = args.dfo_fake_root or args.df0_fake_root
    dfo_real = args.dfo_real_root or args.df0_real_root
    if dfo_fake and dfo_real:
        print("\nBuilding DFo video list ...")
        dataset_videos["DFo"] = build_nested_image_videos(dfo_fake, dfo_real, "DFo")
    else:
        print("\n[skip] DFo: --df0_fake_root/--dfo_fake_root and --df0_real_root/--dfo_real_root not provided.")

    if args.dfd_fake_root and args.dfd_real_root:
        print("\nBuilding DFD video list ...")
        dataset_videos["DFD"] = build_nested_image_videos(args.dfd_fake_root, args.dfd_real_root, "DFD")
    else:
        print("\n[skip] DFD: --dfd_fake_root / --dfd_real_root not provided.")

    if args.dfdc_fake_root and args.dfdc_real_root:
        print("\nBuilding DFDC video list ...")
        dataset_videos["DFDC"] = build_flat_videos(args.dfdc_fake_root, args.dfdc_real_root, "DFDC")
    else:
        print("\n[skip] DFDC: --dfdc_fake_root / --dfdc_real_root not provided.")

    if args.wdf_fake_root and args.wdf_real_root:
        print("\nBuilding WDF video list ...")
        dataset_videos["WDF"] = build_flat_videos(args.wdf_fake_root, args.wdf_real_root, "WDF")
    else:
        print("\n[skip] WDF: --wdf_fake_root / --wdf_real_root not provided.")

    if args.uadfv_fake_root and args.uadfv_real_root:
        print("\nBuilding UADFV video list ...")
        dataset_videos["UADFV"] = build_nested_image_videos(args.uadfv_fake_root, args.uadfv_real_root, "UADFV")
    else:
        print("\n[skip] UADFV: --uadfv_fake_root / --uadfv_real_root not provided.")
    return dataset_videos


def build_frame_items(args) -> list[dict]:
    dataset_videos = build_dataset_videos(args)
    items = []
    print("\nDataset sampling:")
    for seed_offset, (dataset, videos) in enumerate(sorted(dataset_videos.items()), start=100):
        if not videos:
            continue
        selected_videos = stratified_cap(videos, args.max_videos_per_dataset, args.seed + seed_offset)
        dataset_items = []
        for video_id, paths, label in selected_videos:
            indices = sample_indices(len(paths), args.frames_per_video)
            for i in indices:
                dataset_items.append({
                    "path": paths[int(i)],
                    "label": int(label),
                    "dataset": dataset,
                    "video_id": video_id,
                    "frame_position": int(i),
                })
        dataset_items = cap_frames(dataset_items, args.max_frames_per_dataset, args.seed + seed_offset + 1000)
        real_n = sum(1 for item in dataset_items if item["label"] == 0)
        fake_n = sum(1 for item in dataset_items if item["label"] == 1)
        print(
            f"  {dataset}: videos={len(selected_videos)}/{len(videos)}  "
            f"frames={len(dataset_items)}  real={real_n}  fake={fake_n}"
        )
        items.extend(dataset_items)
    if not items:
        raise ValueError("No frame items found. Provide at least one dataset.")
    return items


def load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class FrameDataset(Dataset):
    def __init__(self, items: list[dict], image_size: int):
        self.items = items
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            image = load_image_tensor(item["path"], self.image_size)
            ok = True
        except Exception:
            image = torch.zeros(3, self.image_size, self.image_size)
            ok = False
        return image, item["label"], item["path"], item["dataset"], item["video_id"], item["frame_position"], ok


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj))) if isinstance(obj, dict) else obj
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_stage1_model(checkpoint: str, device: torch.device):
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ViT()
    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"\nLoaded Stage-1 checkpoint: {ckpt_path}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    return model


def select_features(logits_list, features_list, cls_list, source: str) -> torch.Tensor:
    if source == "last_features":
        return features_list[-1].float()
    if source == "concat_features":
        return torch.cat([feat.float() for feat in features_list], dim=1)
    if source == "last_cls":
        return cls_list[-1].float()
    if source == "concat_cls":
        return torch.cat([cls.float() for cls in cls_list], dim=1)
    if source == "last_logits":
        return logits_list[-1].float()
    raise ValueError(f"Unknown feature source: {source}")


@torch.inference_mode()
def extract_frame_embeddings(model, loader, device: torch.device, args):
    embeddings, labels, probs = [], [], []
    paths, datasets, video_ids, frame_positions, ok_flags = [], [], [], [], []
    amp_enabled = device.type == "cuda" and not args.no_amp
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)

    model.eval()
    with autocast_ctx:
        for images, batch_labels, batch_paths, batch_dsets, batch_vids, batch_pos, batch_ok in tqdm(loader, desc="Extracting frame embeddings"):
            images = images.to(device, non_blocking=True)
            logits_list, features_list, cls_list = model(images)
            emb = select_features(logits_list, features_list, cls_list, args.feature_source)
            prob = torch.softmax(logits_list[-1].float(), dim=1)[:, 1]

            embeddings.append(emb.float().cpu().numpy())
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probs.extend(prob.cpu().numpy().tolist())
            paths.extend(list(batch_paths))
            datasets.extend(list(batch_dsets))
            video_ids.extend(list(batch_vids))
            frame_positions.extend(batch_pos.numpy().astype(int).tolist())
            ok_flags.extend(batch_ok.numpy().astype(bool).tolist())

    embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1), dtype=np.float32)
    return (
        embeddings,
        np.asarray(labels, dtype=np.int64),
        np.asarray(probs, dtype=np.float32),
        paths,
        datasets,
        video_ids,
        np.asarray(frame_positions, dtype=np.int64),
        np.asarray(ok_flags, dtype=bool),
    )


def outlier_group_keys(labels: np.ndarray, dataset_tags: List[str], mode: str):
    tags = np.asarray(dataset_tags)
    if mode == "class":
        return np.asarray([f"label={label}" for label in labels])
    if mode == "dataset":
        return tags.astype(str)
    if mode == "dataset_class":
        return np.asarray([f"{dset}/label={label}" for dset, label in zip(tags, labels)])
    return np.asarray(["all"] * len(labels))


def remove_embedding_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    paths: List[str],
    dataset_tags: List[str],
    video_ids: List[str],
    frame_positions: np.ndarray,
    ok_flags: np.ndarray,
    std_factor: float,
    group_mode: str,
):
    if std_factor <= 0:
        print("\nOutlier removal disabled.")
        return embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags

    keys = outlier_group_keys(labels, dataset_tags, group_mode)
    keep = np.ones(len(labels), dtype=bool)
    print(f"\nOutlier removal: group={group_mode}, threshold=mean+{std_factor:.2f}*std")
    for key in sorted(set(keys.tolist())):
        idx = np.where(keys == key)[0]
        if len(idx) < 8:
            print(f"  {key:<22} n={len(idx):>6} removed=0 (too small)")
            continue
        group_emb = embeddings[idx].astype(np.float64)
        centroid = group_emb.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(group_emb - centroid, axis=1)
        threshold = distances.mean() + std_factor * distances.std()
        group_keep = distances <= threshold
        keep[idx] = group_keep
        print(f"  {key:<22} n={len(idx):>6} removed={int((~group_keep).sum()):>5} thr={threshold:.4f}")

    print(f"Total outliers removed: {int((~keep).sum())}/{len(labels)}")
    if keep.all():
        return embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags
    return (
        embeddings[keep],
        labels[keep],
        probs[keep],
        [p for p, k in zip(paths, keep) if k],
        [d for d, k in zip(dataset_tags, keep) if k],
        [v for v, k in zip(video_ids, keep) if k],
        frame_positions[keep],
        ok_flags[keep],
    )


def patch_coverage_for_numba():
    os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
    try:
        import coverage
    except Exception:
        return
    coverage_types = getattr(coverage, "types", None)
    if coverage_types is not None and not hasattr(coverage_types, "Tracer") and hasattr(coverage_types, "TTracer"):
        coverage_types.Tracer = coverage_types.TTracer


def output_with_suffix(out_path: str, suffix: str) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def set_tight_limits(ax, points: np.ndarray, pad_frac: float = 0.06):
    if points.size == 0:
        return
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    ax.set_xlim(x_min - pad_frac * x_span, x_max + pad_frac * x_span)
    ax.set_ylim(y_min - pad_frac * y_span, y_max + pad_frac * y_span)


def plot_umap(coords: np.ndarray, labels: np.ndarray, probs: np.ndarray, dataset_tags: List[str], out_path: str):
    import matplotlib.pyplot as plt

    dataset_tags = np.asarray(dataset_tags)
    preferred_order = ["FFPP", "CDFv2", "CDFv3", "WDF", "UADFV", "DFo", "DFD", "DFDC"]
    present = set(dataset_tags.tolist())
    datasets = [d for d in preferred_order if d in present]
    datasets.extend(sorted(present - set(datasets)))
    dataset_colors = {
        "FFPP": "#0057FF",
        "CDFv2": "#E31A1C",
        "CDFv3": "#00A651",
        "WDF": "#000000",
        "UADFV": "#FFD700",
        "DFo": "#FF4FB3",
        "DFD": "#7B2CBF",
        "DFDC": "#FF8C00",
    }
    fallback_colors = plt.get_cmap("tab20").colors
    colors = {d: dataset_colors.get(d, fallback_colors[i % len(fallback_colors)]) for i, d in enumerate(datasets)}
    saved = []

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    for dset in datasets:
        for label, marker, name in [(0, "o", "Real"), (1, "^", "Fake")]:
            mask = (dataset_tags == dset) & (labels == label)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[dset], marker=marker, s=7, alpha=0.78,
                linewidths=0.05, edgecolors="black", label=f"{dset} - {name}",
            )
    ax.set_title("UMAP of Stage-1 frame embeddings")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=7, loc="best", markerscale=1.4)
    set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)

    split_path = output_with_suffix(out_path, "real_fake_split")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, label, title in [(axes[0], 0, "Real frames"), (axes[1], 1, "Fake frames")]:
        panel_mask = labels == label
        for dset in datasets:
            mask = panel_mask & (dataset_tags == dset)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[dset], s=6, alpha=0.80,
                linewidths=0.04, edgecolors="black", label=dset,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP dim 1")
        ax.set_ylabel("UMAP dim 2")
        ax.legend(fontsize=7, loc="best", markerscale=1.4)
        set_tight_limits(ax, coords[panel_mask])
    fig.suptitle("Stage-1 frame embeddings split by class")
    fig.tight_layout()
    fig.savefig(split_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(split_path)

    label_path = output_with_suffix(out_path, "label")
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    for label, color, marker, name in [(0, "#2E7D32", "o", "Real"), (1, "#C62828", "^", "Fake")]:
        mask = labels == label
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, marker=marker, s=7, alpha=0.78,
                   linewidths=0.05, edgecolors="black", label=name)
    ax.set_title("UMAP of Stage-1 frame embeddings\ncolor = real/fake")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=9, loc="best")
    set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(label_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(label_path)

    prob_path = output_with_suffix(out_path, "prob")
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=probs, cmap="coolwarm", vmin=0, vmax=1,
                    s=7, alpha=0.80, linewidths=0.05, edgecolors="black")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Stage-1 P(fake), layer 23")
    ax.set_title("UMAP of Stage-1 frame embeddings\ncolor = predicted P(fake)")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    set_tight_limits(ax, coords)
    fig.tight_layout()
    fig.savefig(prob_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(prob_path)
    return saved


def load_cached(path: str):
    data = np.load(path, allow_pickle=True)
    return (
        data["embeddings"],
        data["labels"].astype(int),
        data["probs"].astype(float),
        data["paths"].astype(str).tolist(),
        data["dataset_tags"].astype(str).tolist(),
        data["video_ids"].astype(str).tolist(),
        data["frame_positions"].astype(int),
        data["ok_flags"].astype(bool) if "ok_flags" in data else np.ones(len(data["labels"]), dtype=bool),
    )


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("=" * 88)
    print("UMAP of Stage-1 frame-level embeddings")
    print("=" * 88)
    print(f"Device        : {device}")
    print(f"Feature source: {args.feature_source}")
    print(f"Output        : {args.out}")

    if args.embeddings_npz:
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = load_cached(args.embeddings_npz)
        print(f"Loaded cached embeddings: {embeddings.shape} from {args.embeddings_npz}")
    else:
        if not args.checkpoint:
            raise ValueError("Supply --checkpoint or --embeddings_npz.")

        items = build_frame_items(args)
        print(f"\nTotal sampled frames: {len(items)}")
        dataset = FrameDataset(items, args.image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = load_stage1_model(args.checkpoint, device)
        if not args.no_compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile ...")
            model = torch.compile(model)

        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = extract_frame_embeddings(
            model, loader, device, args
        )
        print(f"Extracted embeddings: {embeddings.shape}")
        print(f"Bad/blank image fallbacks: {int((~ok_flags).sum())}")

        cache_path = args.embeddings_out or default_embeddings_path(args.out)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            labels=labels,
            probs=probs,
            paths=np.asarray(paths),
            dataset_tags=np.asarray(dataset_tags),
            video_ids=np.asarray(video_ids),
            frame_positions=frame_positions,
            ok_flags=ok_flags,
            feature_source=np.asarray([args.feature_source]),
        )
        print(f"Cached embeddings -> {cache_path}")

    print_dataset_counts("Frames loaded before outlier removal:", labels, dataset_tags)
    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = remove_embedding_outliers(
        embeddings,
        labels,
        probs,
        paths,
        dataset_tags,
        video_ids,
        frame_positions,
        ok_flags,
        args.outlier_std,
        args.outlier_group,
    )
    print_dataset_counts("Frames used for UMAP after outlier removal:", labels, dataset_tags)

    from sklearn.preprocessing import StandardScaler
    patch_coverage_for_numba()
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install UMAP with: pip install umap-learn") from exc
    except AttributeError as exc:
        raise RuntimeError(
            "UMAP import failed inside numba/coverage. The embeddings are cached; "
            "rerun with --embeddings_npz <cache>, or fix with: pip install -U numba coverage umap-learn"
        ) from exc

    print("\nStandardizing embeddings and running joint UMAP ...")
    scaled = StandardScaler().fit_transform(embeddings)
    n_neighbors = min(args.umap_neighbors, max(2, scaled.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
    )
    coords = reducer.fit_transform(scaled)

    saved = plot_umap(coords, labels, probs, dataset_tags, args.out)
    out_path = Path(args.out)
    coords_path = str(out_path.with_name(f"{out_path.stem}_coords.csv"))
    pd.DataFrame({
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "label": labels,
        "prob_fake": probs,
        "dataset": dataset_tags,
        "video_id": video_ids,
        "frame_position": frame_positions,
        "path": paths,
    }).to_csv(coords_path, index=False)
    saved.append(coords_path)

    print("\nSaved outputs:")
    for path in saved:
        print(f"  {path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
