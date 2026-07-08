"""
Centroid-distance analysis for video-level embeddings from all baseline models.

This script extracts sampled frame embeddings, mean-pools them into video
embeddings, then measures whether real videos are more cross-dataset aligned
than fake videos. It writes a long-format CSV with pairwise centroid distances,
within-centroid dispersion, and nearest-dataset-centroid summaries.

Example:
python centroid_distance_analysis_all_models.py \
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
    --xception_checkpoint checkpoints_xception_ffpp/best.pth \
    --effnet_checkpoint checkpoints_efficientnet_all/best.pth \
    --clip_checkpoint checkpoints_clip_vit_b16_all/best.pth \
    --stage1_checkpoint checkpoints_vit_4layers/best.pth \
    --frames_per_video 4 \
    --out centroid_distance_findings.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


CLASS_NAMES = {0: "real", 1: "fake"}
xception_umap = None
effnet_umap = None
clip_umap = None
stage1_umap = None


def load_project_modules():
    global xception_umap, effnet_umap, clip_umap, stage1_umap
    if xception_umap is not None:
        return
    import umap_clip_vit_b16_frame_embeddings as _clip_umap
    import umap_efficientnet_frame_embeddings as _effnet_umap
    import umap_stage1_frame_embeddings as _stage1_umap
    import umap_xception_frame_embeddings as _xception_umap

    xception_umap = _xception_umap
    effnet_umap = _effnet_umap
    clip_umap = _clip_umap
    stage1_umap = _stage1_umap


def parse_args():
    p = argparse.ArgumentParser(description="Centroid-distance analysis over video-level embeddings.")

    # Checkpoints.
    p.add_argument("--xception_checkpoint", default="checkpoints_xception_ffpp/best.pth")
    p.add_argument("--xception_model_name", default="", help="Override model name stored in checkpoint.")
    p.add_argument("--effnet_checkpoint", default="checkpoints_efficientnet_all/best.pth")
    p.add_argument("--effnet_model_name", default="", help="Override model name stored in checkpoint.")
    p.add_argument("--clip_checkpoint", default="checkpoints_clip_vit_b16_all/best.pth")
    p.add_argument("--clip_model_name", default="", help="Override model name stored in checkpoint.")
    p.add_argument("--stage1_checkpoint", default="checkpoints_vit_4layers/best.pth")
    p.add_argument(
        "--stage1_feature_source",
        default="last_features",
        choices=["last_features", "concat_features", "last_cls", "concat_cls", "last_logits"],
    )
    p.add_argument("--skip_missing_checkpoints", action="store_true", default=True)
    p.add_argument("--strict_checkpoints", action="store_true", help="Fail instead of skipping missing checkpoints.")

    # Runtime.
    p.add_argument("--out", default="centroid_distance_findings.csv")
    p.add_argument("--cache_dir", default="centroid_embedding_cache")
    p.add_argument("--refresh_cache", action="store_true", help="Ignore existing valid caches.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true", help="Only affects Stage-1 model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_videos_per_centroid", type=int, default=2)

    # Sampling shared by all models.
    p.add_argument("--frames_per_video", type=int, default=4, help="0 = use all frames from selected videos.")
    p.add_argument("--max_videos_per_dataset", type=int, default=0, help="0 = no video cap.")
    p.add_argument("--max_frames_per_dataset", type=int, default=0, help="0 = no per-dataset frame cap.")
    p.add_argument("--max_total_frames", type=int, default=0, help="0 = no global frame cap.")
    p.add_argument("--val_ratio", type=float, default=0.0, help="FF++ val split ratio. 0 = all FF++ videos.")

    # Image sizes.
    p.add_argument("--xception_image_size", type=int, default=299)
    p.add_argument("--effnet_image_size", type=int, default=380)
    p.add_argument("--clip_image_size", type=int, default=224)
    p.add_argument("--stage1_image_size", type=int, default=256)

    # FF++.
    p.add_argument("--manifest", default="")
    p.add_argument("--root_dir", default="")

    # CDFv2.
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # CDFv3 / CDF++.
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # DFo / DeeperForensics-1.0.
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")

    # DFD.
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC.
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")

    # WDF.
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV.
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")
    return p.parse_args()


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def checkpoint_path(path: str) -> Path:
    ckpt = Path(path).expanduser()
    if ckpt.is_dir():
        ckpt = ckpt / "best.pth"
    return ckpt


def items_signature(items: List[dict], extra: dict) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    for item in items:
        h.update(str(item["dataset"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(item["video_id"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(item["label"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(item["frame_position"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(item["path"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def save_cache(path: Path, arrays: tuple, metadata: dict):
    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=embeddings,
        labels=labels,
        probs=probs,
        paths=np.asarray(paths),
        dataset_tags=np.asarray(dataset_tags),
        video_ids=np.asarray(video_ids),
        frame_positions=frame_positions,
        ok_flags=ok_flags,
        metadata=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )


def load_cache(path: Path, expected_metadata: dict):
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    try:
        metadata = json.loads(str(data["metadata"][0]))
    except Exception:
        return None
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            print(f"  cache mismatch for {path.name}: {key} differs")
            return None
    print(f"  loaded cache: {path}")
    arrays = (
        data["embeddings"],
        data["labels"].astype(int),
        data["probs"].astype(float),
        data["paths"].astype(str).tolist(),
        data["dataset_tags"].astype(str).tolist(),
        data["video_ids"].astype(str).tolist(),
        data["frame_positions"].astype(int),
        data["ok_flags"].astype(bool),
    )
    return arrays, metadata


def extract_timm_like(
    model_key: str,
    items: List[dict],
    image_size: int,
    load_model_fn: Callable,
    extract_fn: Callable,
    checkpoint: str,
    model_name: str,
    device: torch.device,
    args,
):
    model, resolved_model_name = load_model_fn(checkpoint, model_name, device)
    loader = DataLoader(
        xception_umap.FrameDataset(items, image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    extract_args = SimpleNamespace(no_amp=args.no_amp)
    arrays = extract_fn(model, loader, device, extract_args)
    return arrays, resolved_model_name


def extract_stage1(
    items: List[dict],
    image_size: int,
    checkpoint: str,
    device: torch.device,
    args,
):
    ckpt_path = checkpoint_path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = stage1_umap.ViT()
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    state = stage1_umap.clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"\nLoaded Stage-1 checkpoint: {ckpt_path}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    if not args.no_compile and hasattr(torch, "compile"):
        print("  compiling Stage-1 model with torch.compile ...")
        model = torch.compile(model)
    loader = DataLoader(
        stage1_umap.FrameDataset(items, image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    extract_args = SimpleNamespace(no_amp=args.no_amp, feature_source=args.stage1_feature_source)
    arrays = stage1_umap.extract_frame_embeddings(model, loader, device, extract_args)
    return arrays, f"frame_model.ViT/{args.stage1_feature_source}"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def distance_matrix(a: np.ndarray, b: np.ndarray, space: str) -> np.ndarray:
    if space == "cosine_normalized":
        return 1.0 - np.clip(a @ b.T, -1.0, 1.0)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def add_row(rows: List[dict], model_info: dict, metric: str, space: str, class_name: str = "",
            value: float = math.nan, dataset: str = "", dataset_a: str = "", dataset_b: str = "",
            n: int = 0, n_a: int = 0, n_b: int = 0, note: str = ""):
    rows.append({
        "model_key": model_info["model_key"],
        "model_name": model_info["model_name"],
        "checkpoint": model_info["checkpoint"],
        "feature_source": model_info["feature_source"],
        "space": space,
        "metric": metric,
        "class": class_name,
        "dataset": dataset,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "n": int(n),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "value": float(value) if value is not None and not pd.isna(value) else np.nan,
        "note": note,
    })


def compute_rows_for_space(
    rows: List[dict],
    model_info: dict,
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    dataset_tags: List[str],
    space: str,
    min_videos: int,
):
    tags = np.asarray(dataset_tags)
    if space == "cosine_normalized":
        x = l2_normalize(embeddings.astype(np.float64))
    elif space == "zscore_l2":
        x = zscore(embeddings.astype(np.float64))
    else:
        raise ValueError(f"Unknown space: {space}")

    add_row(rows, model_info, "video_count_total", space, value=len(labels), n=len(labels))
    for label_value, class_name in CLASS_NAMES.items():
        class_mask = labels == label_value
        add_row(rows, model_info, "video_count_class", space, class_name, value=int(class_mask.sum()), n=int(class_mask.sum()))
        datasets = sorted(set(tags[class_mask].tolist()))
        centroids: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}

        for dset in datasets:
            mask = class_mask & (tags == dset)
            n = int(mask.sum())
            add_row(rows, model_info, "video_count_dataset_class", space, class_name, value=n, dataset=dset, n=n)
            if n >= min_videos:
                centroids[dset] = x[mask].mean(axis=0)
                counts[dset] = n
                dists = distance_matrix(x[mask], centroids[dset][None, :], space).reshape(-1)
                add_row(
                    rows, model_info, "mean_within_centroid_distance", space, class_name,
                    value=float(dists.mean()), dataset=dset, n=n,
                    note="Mean distance of videos to their own dataset/class centroid.",
                )

        if len(centroids) < 2:
            add_row(
                rows, model_info, "mean_pairwise_centroid_distance", space, class_name,
                value=np.nan, dataset="ALL", n=int(class_mask.sum()),
                note=f"Fewer than two datasets had >= {min_videos} videos for this class.",
            )
            continue

        pair_distances = []
        centroid_names = sorted(centroids)
        centroid_mat = np.stack([centroids[name] for name in centroid_names], axis=0)
        centroid_dist = distance_matrix(centroid_mat, centroid_mat, space)
        for i, dset_a in enumerate(centroid_names):
            for j in range(i + 1, len(centroid_names)):
                dset_b = centroid_names[j]
                dist = float(centroid_dist[i, j])
                pair_distances.append(dist)
                add_row(
                    rows, model_info, "pairwise_centroid_distance", space, class_name,
                    value=dist, dataset_a=dset_a, dataset_b=dset_b,
                    n_a=counts[dset_a], n_b=counts[dset_b],
                    note="Distance between dataset centroids within the same class.",
                )

        pair_distances = np.asarray(pair_distances, dtype=np.float64)
        add_row(
            rows, model_info, "mean_pairwise_centroid_distance", space, class_name,
            value=float(pair_distances.mean()), dataset="ALL", n=int(class_mask.sum()),
            note="Lower value means dataset centroids for this class are more aligned.",
        )
        add_row(
            rows, model_info, "std_pairwise_centroid_distance", space, class_name,
            value=float(pair_distances.std()), dataset="ALL", n=int(class_mask.sum()),
        )

        usable = class_mask & np.isin(tags, centroid_names)
        x_class = x[usable]
        tag_class = tags[usable]
        label_class = labels[usable]
        dist_to_centroids = distance_matrix(x_class, centroid_mat, space)
        nearest_idx = dist_to_centroids.argmin(axis=1)
        nearest_dataset = np.asarray([centroid_names[i] for i in nearest_idx])
        centroid_acc = float((nearest_dataset == tag_class).mean()) if len(tag_class) else np.nan
        own_idx = np.asarray([centroid_names.index(d) for d in tag_class], dtype=int)
        own_dist = dist_to_centroids[np.arange(len(tag_class)), own_idx]
        dist_to_other = dist_to_centroids.copy()
        dist_to_other[np.arange(len(tag_class)), own_idx] = np.inf
        nearest_other_dist = dist_to_other.min(axis=1)
        margin = nearest_other_dist - own_dist
        add_row(
            rows, model_info, "nearest_dataset_centroid_accuracy", space, class_name,
            value=centroid_acc, dataset="ALL", n=int(len(label_class)),
            note="Higher value means video embeddings are more dataset-identifiable within this class.",
        )
        add_row(
            rows, model_info, "mean_dataset_centroid_margin", space, class_name,
            value=float(np.mean(margin)), dataset="ALL", n=int(len(label_class)),
            note="Nearest other dataset centroid distance minus own centroid distance.",
        )


def compute_findings(model_info: dict, video_arrays: tuple, min_videos: int) -> List[dict]:
    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_counts, ok_flags = video_arrays
    rows: List[dict] = []
    for space in ("cosine_normalized", "zscore_l2"):
        compute_rows_for_space(rows, model_info, embeddings, labels, probs, dataset_tags, space, min_videos)

    # Convenient high-level fake-vs-real ratios for the two summary metrics.
    df = pd.DataFrame(rows)
    for space in ("cosine_normalized", "zscore_l2"):
        for metric in ("mean_pairwise_centroid_distance", "nearest_dataset_centroid_accuracy", "mean_dataset_centroid_margin"):
            sub = df[(df["space"] == space) & (df["metric"] == metric) & (df["dataset"] == "ALL")]
            values = {row["class"]: row["value"] for _, row in sub.iterrows()}
            real = values.get("real", np.nan)
            fake = values.get("fake", np.nan)
            if pd.notna(real) and pd.notna(fake):
                add_row(
                    rows, model_info, f"fake_minus_real_{metric}", space,
                    value=float(fake - real), dataset="ALL",
                    note="Positive value means the metric is larger for fake videos than real videos.",
                )
                if abs(real) > 1e-12:
                    add_row(
                        rows, model_info, f"fake_over_real_{metric}", space,
                        value=float(fake / real), dataset="ALL",
                        note="Ratio > 1 means the metric is larger for fake videos than real videos.",
                    )
    return rows


def model_configs(args):
    return [
        {
            "model_key": "xception",
            "checkpoint": args.xception_checkpoint,
            "image_size": args.xception_image_size,
            "feature_source": "penultimate",
            "extract": lambda items, device: extract_timm_like(
                "xception", items, args.xception_image_size,
                xception_umap.load_model, xception_umap.extract_embeddings,
                args.xception_checkpoint, args.xception_model_name, device, args,
            ),
        },
        {
            "model_key": "efficientnet",
            "checkpoint": args.effnet_checkpoint,
            "image_size": args.effnet_image_size,
            "feature_source": "penultimate",
            "extract": lambda items, device: extract_timm_like(
                "efficientnet", items, args.effnet_image_size,
                effnet_umap.load_model, xception_umap.extract_embeddings,
                args.effnet_checkpoint, args.effnet_model_name, device, args,
            ),
        },
        {
            "model_key": "clip_vit_b16",
            "checkpoint": args.clip_checkpoint,
            "image_size": args.clip_image_size,
            "feature_source": "pre_logits",
            "extract": lambda items, device: extract_timm_like(
                "clip_vit_b16", items, args.clip_image_size,
                clip_umap.load_model, clip_umap.extract_embeddings,
                args.clip_checkpoint, args.clip_model_name, device, args,
            ),
        },
        {
            "model_key": "stage1_vit_4layers",
            "checkpoint": args.stage1_checkpoint,
            "image_size": args.stage1_image_size,
            "feature_source": args.stage1_feature_source,
            "extract": lambda items, device: extract_stage1(
                items, args.stage1_image_size, args.stage1_checkpoint, device, args,
            ),
        },
    ]


def main():
    args = parse_args()
    if args.strict_checkpoints:
        args.skip_missing_checkpoints = False
    load_project_modules()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("Centroid-distance analysis over video-level embeddings")
    print("=" * 88)
    print(f"Device      : {device}")
    print(f"Output CSV  : {args.out}")
    print(f"Cache dir   : {args.cache_dir}")

    items = xception_umap.build_frame_items(args)
    print(f"\nTotal sampled frames shared by all models: {len(items)}")
    shared_sig = items_signature(items, {
        "frames_per_video": args.frames_per_video,
        "max_videos_per_dataset": args.max_videos_per_dataset,
        "max_frames_per_dataset": args.max_frames_per_dataset,
        "max_total_frames": args.max_total_frames,
        "val_ratio": args.val_ratio,
    })

    all_rows = []
    cache_dir = Path(args.cache_dir)
    for config in model_configs(args):
        model_key = config["model_key"]
        ckpt = checkpoint_path(config["checkpoint"])
        print("\n" + "=" * 88)
        print(f"Model: {model_key}")
        print("=" * 88)
        print(f"Checkpoint: {ckpt}")
        if not ckpt.is_file():
            message = f"Missing checkpoint for {model_key}: {ckpt}"
            if args.skip_missing_checkpoints:
                print(f"  SKIP: {message}")
                continue
            raise FileNotFoundError(message)

        metadata = {
            "model_key": model_key,
            "checkpoint": str(ckpt),
            "items_signature": shared_sig,
            "image_size": int(config["image_size"]),
            "feature_source": str(config["feature_source"]),
        }
        cache_path = cache_dir / f"{safe_key(model_key)}_{safe_key(ckpt.stem)}_frame_embeddings.npz"
        cached = None if args.refresh_cache else load_cache(cache_path, metadata)
        arrays = None
        cache_metadata = {}
        if cached is not None:
            arrays, cache_metadata = cached
        resolved_model_name = model_key
        if arrays is None:
            print("  extracting frame embeddings ...")
            arrays, resolved_model_name = config["extract"](items, device)
            save_meta = dict(metadata)
            save_meta["resolved_model_name"] = resolved_model_name
            save_cache(cache_path, arrays, save_meta)
            print(f"  cached frame embeddings -> {cache_path}")
        else:
            resolved_model_name = cache_metadata.get("resolved_model_name", model_key)

        video_arrays = xception_umap.aggregate_frames_to_videos(*arrays)
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_counts, ok_flags = video_arrays
        print(f"  video embeddings: {embeddings.shape}")
        xception_umap.print_frame_counts("  Videos available for centroid analysis:", labels, dataset_tags)
        xception_umap.print_sampled_auc("  Video-level performance from mean frame probabilities:", labels, probs, dataset_tags)

        model_info = {
            "model_key": model_key,
            "model_name": resolved_model_name,
            "checkpoint": str(ckpt),
            "feature_source": str(config["feature_source"]),
        }
        all_rows.extend(compute_findings(model_info, video_arrays, args.min_videos_per_centroid))

    if not all_rows:
        raise RuntimeError("No model findings were produced. Check checkpoint paths and dataset arguments.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    sort_cols = ["model_key", "space", "metric", "class", "dataset", "dataset_a", "dataset_b"]
    df = df.sort_values(sort_cols, kind="stable")
    df.to_csv(out_path, index=False)

    summary = df[
        df["metric"].isin([
            "fake_minus_real_mean_pairwise_centroid_distance",
            "fake_over_real_mean_pairwise_centroid_distance",
            "fake_minus_real_nearest_dataset_centroid_accuracy",
            "fake_minus_real_mean_dataset_centroid_margin",
        ])
    ][["model_key", "space", "metric", "value"]]

    print("\nKey summary rows:")
    if len(summary):
        print(summary.to_string(index=False))
    else:
        print("  No summary rows available.")
    print("\nSaved:")
    print(f"  {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
