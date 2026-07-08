"""
UMAP real/fake split for a pretrained Xception frame model, aggregated to videos.

This loads checkpoints produced by train_xception_ffpp.py, extracts the
penultimate Xception feature for sampled frames, mean-pools them per video, and
saves only a real-vs-fake split UMAP.

Example:
python umap_xception_frame_embeddings.py \
    --checkpoint checkpoints_xception_ffpp/best.pth \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out \
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --cdfv3_root /media/tarun/B482367C823642E2/usr/cdfv3_face_crops \
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
    --max_frames_per_dataset 2000 \
    --out umap_xception_real_fake_split.png
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Real/fake split UMAP for Xception frame embeddings.")
    p.add_argument("--checkpoint", default="", help="Xception checkpoint from train_xception_ffpp.py.")
    p.add_argument("--embeddings_npz", default="", help="Cached embeddings from this script.")
    p.add_argument("--embeddings_out", default="", help="Cache output path. Default: next to --out.")
    p.add_argument("--out", default="umap_xception_real_fake_split.png")
    p.add_argument("--model_name", default="", help="Override checkpoint model_name. Usually leave empty.")
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--frames_per_video", type=int, default=4, help="0 = use all frames from selected videos.")
    p.add_argument("--max_videos_per_dataset", type=int, default=0, help="0 = no video cap.")
    p.add_argument("--max_frames_per_dataset", type=int, default=0, help="0 = no per-dataset frame cap.")
    p.add_argument("--max_total_frames", type=int, default=0, help="0 = no global frame cap after dataset sampling.")
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

    # DFo / DeeperForensics-1.0
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")

    # DFD
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")

    # WDF
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")

    # UMAP
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


def cap_items(items: list[dict], cap: int, seed: int):
    if cap <= 0 or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    reals = [item for item in items if item["label"] == 0]
    fakes = [item for item in items if item["label"] == 1]
    if not reals or not fakes:
        idx = rng.choice(len(items), size=cap, replace=False)
        return [items[int(i)] for i in sorted(idx)]

    n_real = min(len(reals), cap // 2)
    n_fake = min(len(fakes), cap - n_real)
    n_real = min(len(reals), cap - n_fake)
    real_idx = rng.choice(len(reals), size=n_real, replace=False)
    fake_idx = rng.choice(len(fakes), size=n_fake, replace=False)
    capped = [reals[int(i)] for i in real_idx] + [fakes[int(i)] for i in fake_idx]
    order = rng.permutation(len(capped))
    return [capped[int(i)] for i in order]


def video_id_from_ffpp_sample(sample_dir: str) -> str:
    parts = Path(str(sample_dir).replace("\\", "/")).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = basename.rfind("_frame_")
    if idx != -1:
        clip_id = basename[:idx]
    else:
        match = re.search(r"_f\d+$", basename)
        clip_id = basename[:match.start()] if match else basename
    return f"{prefix}/{clip_id}" if prefix else clip_id


def print_video_counts(name: str, videos: List[Tuple[str, List[str], int]], extra: str = ""):
    real_n = sum(1 for _, _, label in videos if label == 0)
    fake_n = sum(1 for _, _, label in videos if label == 1)
    suffix = f" [{extra}]" if extra else ""
    print(f"  [{name}{suffix}] {len(videos)} videos  (real={real_n}, fake={fake_n})")


def build_ffpp_videos(manifest_csv: str, root_dir: str, val_ratio: float) -> List[Tuple[str, List[str], int]]:
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(video_id_from_ffpp_sample)

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


def build_dataset_videos(args):
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
    print("\nDataset/frame sampling:")
    for seed_offset, (dataset, videos) in enumerate(sorted(dataset_videos.items()), start=100):
        if not videos:
            continue
        selected_videos = stratified_cap(videos, args.max_videos_per_dataset, args.seed + seed_offset)
        dataset_items = []
        for video_id, paths, label in selected_videos:
            for i in sample_indices(len(paths), args.frames_per_video):
                dataset_items.append({
                    "path": paths[int(i)],
                    "label": int(label),
                    "dataset": dataset,
                    "video_id": video_id,
                    "frame_position": int(i),
                })
        dataset_items = cap_items(dataset_items, args.max_frames_per_dataset, args.seed + seed_offset + 1000)
        real_n = sum(1 for item in dataset_items if item["label"] == 0)
        fake_n = sum(1 for item in dataset_items if item["label"] == 1)
        print(
            f"  {dataset}: videos={len(selected_videos)}/{len(videos)}  "
            f"frames={len(dataset_items)}  real={real_n}  fake={fake_n}"
        )
        items.extend(dataset_items)
    items = cap_items(items, args.max_total_frames, args.seed + 9999)
    if not items:
        raise ValueError("No frames found. Provide at least one dataset.")
    return items


class FrameDataset(Dataset):
    def __init__(self, items: list[dict], image_size: int):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item["path"]).convert("RGB")
            img = ImageOps.exif_transpose(img)
            image = self.tf(img)
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


def build_xception(model_name: str):
    import timm

    try:
        return timm.create_model(model_name, pretrained=False, num_classes=2)
    except Exception:
        if model_name == "xception":
            print("Could not create timm model 'xception'. Retrying 'legacy_xception'.")
            return timm.create_model("legacy_xception", pretrained=False, num_classes=2)
        raise


def load_model(checkpoint: str, model_name_override: str, device: torch.device):
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_name = raw.get("model_name", "xception") if isinstance(raw, dict) else "xception"
    model_name = model_name_override or checkpoint_model_name
    model = build_xception(model_name)
    state = clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"  model_name     : {model_name}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    return model.to(device).eval(), model_name


def forward_features_and_logits(model, images: torch.Tensor):
    features = model.forward_features(images)
    emb = None
    logits = None

    if hasattr(model, "forward_head"):
        try:
            emb = model.forward_head(features, pre_logits=True)
        except TypeError:
            emb = None
        try:
            logits = model.forward_head(features)
        except TypeError:
            logits = None

    if emb is None:
        if features.dim() == 4:
            if hasattr(model, "global_pool"):
                emb = model.global_pool(features)
            else:
                emb = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            emb = features[:, 0]
        else:
            emb = features.flatten(1)
    if emb.dim() > 2:
        emb = emb.flatten(1)

    if logits is None:
        classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
        if classifier is not None and not isinstance(classifier, torch.nn.Identity):
            logits = classifier(emb)
        else:
            logits = model(images)
    if emb.dim() > 2:
        emb = emb.flatten(1)
    return emb.float(), logits.float()


@torch.inference_mode()
def extract_embeddings(model, loader, device: torch.device, args):
    embeddings, labels, probs = [], [], []
    paths, datasets, video_ids, frame_positions, ok_flags = [], [], [], [], []
    amp_enabled = device.type == "cuda" and not args.no_amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)

    with autocast:
        for images, batch_labels, batch_paths, batch_dsets, batch_vids, batch_pos, batch_ok in tqdm(loader, desc="Extracting Xception features"):
            images = images.to(device, non_blocking=True)
            emb, logits = forward_features_and_logits(model, images)
            prob = torch.softmax(logits, dim=1)[:, 1]
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probs.extend(prob.cpu().numpy().tolist())
            paths.extend(list(batch_paths))
            datasets.extend(list(batch_dsets))
            video_ids.extend(list(batch_vids))
            frame_positions.extend(batch_pos.numpy().astype(int).tolist())
            ok_flags.extend(batch_ok.numpy().astype(bool).tolist())

    return (
        np.concatenate(embeddings, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(probs, dtype=np.float32),
        paths,
        datasets,
        video_ids,
        np.asarray(frame_positions, dtype=np.int64),
        np.asarray(ok_flags, dtype=bool),
    )


def print_frame_counts(title: str, labels: np.ndarray, dataset_tags: List[str]):
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


def print_sampled_auc(title: str, labels: np.ndarray, probs: np.ndarray, dataset_tags: List[str]):
    tags = np.asarray(dataset_tags)
    print(f"\n{title}")
    print("  dataset      n   real   fake      AUC       AP")
    print("  -----------------------------------------------")

    def row(name: str, mask: np.ndarray):
        n = int(mask.sum())
        real_n = int(((labels == 0) & mask).sum())
        fake_n = int(((labels == 1) & mask).sum())
        if n == 0:
            return
        if real_n == 0 or fake_n == 0:
            print(f"  {name:<10} {n:>5} {real_n:>6} {fake_n:>6}      nan      nan  (single class)")
            return
        auc = roc_auc_score(labels[mask], probs[mask])
        ap = average_precision_score(labels[mask], probs[mask])
        print(f"  {name:<10} {n:>5} {real_n:>6} {fake_n:>6}  {auc:>7.4f}  {ap:>7.4f}")

    for dset in sorted(set(tags.tolist())):
        row(dset, tags == dset)
    row("ALL", np.ones(len(labels), dtype=bool))


def aggregate_frames_to_videos(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    paths: List[str],
    dataset_tags: List[str],
    video_ids: List[str],
    frame_positions: np.ndarray,
    ok_flags: np.ndarray,
):
    groups = {}
    for idx, (dset, vid) in enumerate(zip(dataset_tags, video_ids)):
        groups.setdefault((str(dset), str(vid)), []).append(idx)

    out_embeddings, out_labels, out_probs = [], [], []
    out_paths, out_datasets, out_video_ids, out_frame_counts, out_ok_flags = [], [], [], [], []
    mixed_label_groups = 0
    for (dset, vid), idxs in sorted(groups.items()):
        idx = np.asarray(idxs, dtype=int)
        group_labels = labels[idx].astype(int)
        if len(set(group_labels.tolist())) > 1:
            mixed_label_groups += 1
        label = int(np.round(group_labels.mean()))
        out_embeddings.append(embeddings[idx].mean(axis=0))
        out_probs.append(float(probs[idx].mean()))
        out_labels.append(label)
        out_paths.append(paths[int(idx[0])])
        out_datasets.append(dset)
        out_video_ids.append(vid)
        out_frame_counts.append(int(len(idx)))
        out_ok_flags.append(bool(ok_flags[idx].all()))

    if mixed_label_groups:
        print(f"WARNING: {mixed_label_groups} video groups contained mixed frame labels; using rounded mean label.")
    print(f"\nAggregated frames -> videos: {len(labels)} frames -> {len(out_labels)} videos")
    return (
        np.stack(out_embeddings, axis=0),
        np.asarray(out_labels, dtype=np.int64),
        np.asarray(out_probs, dtype=np.float32),
        out_paths,
        out_datasets,
        out_video_ids,
        np.asarray(out_frame_counts, dtype=np.int64),
        np.asarray(out_ok_flags, dtype=bool),
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


def remove_outliers(embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags, std_factor, group_mode):
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
        group = embeddings[idx].astype(np.float64)
        centroid = group.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(group - centroid, axis=1)
        threshold = distances.mean() + std_factor * distances.std()
        group_keep = distances <= threshold
        keep[idx] = group_keep
        print(f"  {key:<22} n={len(idx):>6} removed={int((~group_keep).sum()):>5} thr={threshold:.4f}")
    print(f"Total outliers removed: {int((~keep).sum())}/{len(labels)}")
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


def set_tight_limits(ax, points: np.ndarray, pad_frac: float = 0.06):
    if points.size == 0:
        return
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    ax.set_xlim(x_min - pad_frac * x_span, x_max + pad_frac * x_span)
    ax.set_ylim(y_min - pad_frac * y_span, y_max + pad_frac * y_span)


def plot_real_fake_split(coords: np.ndarray, labels: np.ndarray, dataset_tags: List[str], out_path: str):
    import matplotlib.pyplot as plt

    dataset_tags = np.asarray(dataset_tags)
    preferred_order = ["FFPP", "CDFv2", "CDFv3", "WDF", "UADFV", "DFo", "DFD", "DFDC"]
    present = set(dataset_tags.tolist())
    datasets = [d for d in preferred_order if d in present]
    datasets.extend(sorted(present - set(datasets)))
    base_colors = {
        "FFPP": "#0057FF",
        "CDFv2": "#E31A1C",
        "CDFv3": "#00A651",
        "WDF": "#000000",
        "UADFV": "#FFD700",
        "DFo": "#FF4FB3",
        "DFD": "#8B5A2B",
        "DFDC": "#FF8C00",
    }
    fallback = plt.get_cmap("tab20").colors
    colors = {d: base_colors.get(d, fallback[i % len(fallback)]) for i, d in enumerate(datasets)}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, label, title in [(axes[0], 0, "Real videos"), (axes[1], 1, "Fake videos")]:
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
    fig.suptitle("Video embeddings split by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    print("UMAP real/fake split for Xception frame embeddings")
    print("=" * 88)
    print(f"Device : {device}")
    print(f"Output : {args.out}")

    if args.embeddings_npz:
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = load_cached(args.embeddings_npz)
        print(f"Loaded cached embeddings: {embeddings.shape} from {args.embeddings_npz}")
    else:
        if not args.checkpoint:
            raise ValueError("Supply --checkpoint or --embeddings_npz.")
        items = build_frame_items(args)
        print(f"\nTotal sampled frames: {len(items)}")
        loader = DataLoader(
            FrameDataset(items, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        model, model_name = load_model(args.checkpoint, args.model_name, device)
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = extract_embeddings(
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
            model_name=np.asarray([model_name]),
        )
        print(f"Cached embeddings -> {cache_path}")

    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = aggregate_frames_to_videos(
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags
    )
    print_frame_counts("Videos before outlier removal:", labels, dataset_tags)
    print_sampled_auc("Video-level performance from mean frame probabilities:", labels, probs, dataset_tags)
    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = remove_outliers(
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags,
        args.outlier_std, args.outlier_group,
    )
    print_frame_counts("Videos used for UMAP:", labels, dataset_tags)
    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 videos for UMAP after filtering.")

    from sklearn.preprocessing import StandardScaler
    patch_coverage_for_numba()
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install UMAP with: pip install umap-learn") from exc
    except AttributeError as exc:
        raise RuntimeError(
            "UMAP import failed inside numba/coverage. Embeddings are cached, so rerun "
            "with --embeddings_npz <cache>, or run: pip install -U numba coverage umap-learn"
        ) from exc

    print("\nStandardizing embeddings and fitting UMAP ...")
    scaled = StandardScaler().fit_transform(embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(args.umap_neighbors, max(2, scaled.shape[0] - 1)),
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
    )
    coords = reducer.fit_transform(scaled)

    plot_real_fake_split(coords, labels, dataset_tags, args.out)
    out_path = Path(args.out)
    coords_path = str(out_path.with_name(f"{out_path.stem}_coords.csv"))
    pd.DataFrame({
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "label": labels,
        "prob_fake": probs,
        "dataset": dataset_tags,
        "video_id": video_ids,
        "n_frames": frame_positions,
        "example_path": paths,
    }).to_csv(coords_path, index=False)
    print("\nSaved outputs:")
    print(f"  {args.out}")
    print(f"  {coords_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
