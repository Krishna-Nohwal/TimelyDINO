"""
Cross-dataset real/fake similarity heatmaps for Stage-2 video embeddings.

This script uses the same VideoViT embedding extraction path as
tsne_predictions1.py, then groups video-level embeddings by dataset and label:

    FFPP Real, FFPP Fake, CDFv2 Real, CDFv2 Fake, ...

It saves two heatmaps:
  1. Prototype cosine similarity between group centroids.
  2. Mean pairwise cosine similarity between videos from each group pair.

The expected paper-facing signal is that real groups from different datasets
are more mutually similar than fake groups or real-fake pairs.

Full commands
-------------
First cache embeddings once with the UMAP script:

python tsne_predictions1.py \
    --checkpoint /home/tarun/Desktop/best/best.pth \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/ \
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --dfo_fake_root /media/tarun/B482367C823642E2/usr/df1.0_faces/fake \
    --dfo_real_root /media/tarun/B482367C823642E2/usr/df1.0_faces/real \
    --wdf_fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --wdf_real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --uadfv_fake_root /media/tarun/B482367C823642E2/usr/uadfv_faces/fake \
    --uadfv_real_root /media/tarun/B482367C823642E2/usr/uadfv_faces/real \
    --num_frames 32 \
    --max_videos_per_dataset 0 \
    --train_real_root /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/real \
    --embeddings_out cached_video_embeddings.npz \
    --out umap_predictions.png

Then create the similarity heatmaps from cached embeddings:

python cross_dataset_similarity_heatmap.py \
    --embeddings_npz cached_video_embeddings.npz \
    --out_dir similarity_heatmaps \
    --prefix cross_dataset
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tsne_predictions1 import (
    VideoViT,
    _dataset_balanced_items,
    _load_frames,
    _sample_frame_indices,
    build_cdfv2_videos,
    build_cdfv3_videos,
    build_ffpp_videos,
    build_nested_image_videos,
    build_wdf_videos,
    build_memory_bank,
    clip_collate_fn,
    extract_embeddings,
    load_model,
)


LABEL_NAMES = {0: "Real", 1: "Fake"}
SHORT_LABEL_NAMES = {0: "R", 1: "F"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-vs-fake cross-dataset cosine similarity heatmaps."
    )
    p.add_argument("--checkpoint", default="",
                   help="Stage-2 frame-end VideoViT checkpoint. Not needed if "
                        "--embeddings_npz is supplied.")
    p.add_argument("--embeddings_npz", default="",
                   help="Optional cached embeddings .npz produced by this script "
                        "or tsne_predictions1.py --embeddings_out.")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_videos_per_dataset", type=int, default=0,
                   help="Equal number of videos to sample from each dataset "
                        "(0 = use the smallest available dataset size).")
    p.add_argument("--no_compile", action="store_true")

    # FF++
    p.add_argument("--manifest", default="",
                   help="FF++ manifest CSV.")
    p.add_argument("--root_dir", default="",
                   help="FF++ frame root dir.")
    p.add_argument("--val_ratio", type=float, default=0.05)

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # DFo / DeeperForensics-1.0 style nested roots
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")

    # WDF flat roots
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV style nested roots
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")

    # CDFv3 / CDF++ optional
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # Memory bank, only needed if checkpoint uses it.
    p.add_argument("--train_real_root", default="")
    p.add_argument("--knn_k", type=int, default=32)
    p.add_argument("--bank_batch_size", type=int, default=16)

    p.add_argument("--out_dir", default="similarity_heatmaps")
    p.add_argument("--prefix", default="cross_dataset")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    return p.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def build_video_items(args) -> List[Tuple[str, List[str], int, str]]:
    dataset_videos = {}

    if args.manifest and args.root_dir:
        print("\nBuilding FF++ video list (val split only) ...")
        dataset_videos["FFPP"] = build_ffpp_videos(
            args.manifest, args.root_dir, args.val_ratio
        )
    else:
        print("\n[skip] FF++: --manifest / --root_dir not provided.")

    if args.cdfv2_fake_root and args.cdfv2_real_root:
        print("\nBuilding CDFv2 video list ...")
        dataset_videos["CDFv2"] = build_cdfv2_videos(
            args.cdfv2_fake_root, args.cdfv2_real_root
        )
    else:
        print("\n[skip] CDFv2: --cdfv2_fake_root / --cdfv2_real_root not provided.")

    if args.dfo_fake_root and args.dfo_real_root:
        print("\nBuilding DFo video list ...")
        dataset_videos["DFo"] = build_nested_image_videos(
            args.dfo_fake_root, args.dfo_real_root, "DFo"
        )
    else:
        print("\n[skip] DFo: --dfo_fake_root / --dfo_real_root not provided.")

    if args.wdf_fake_root and args.wdf_real_root:
        print("\nBuilding WDF video list ...")
        dataset_videos["WDF"] = build_wdf_videos(
            args.wdf_fake_root, args.wdf_real_root
        )
    else:
        print("\n[skip] WDF: --wdf_fake_root / --wdf_real_root not provided.")

    if args.uadfv_fake_root and args.uadfv_real_root:
        print("\nBuilding UADFV video list ...")
        dataset_videos["UADFV"] = build_nested_image_videos(
            args.uadfv_fake_root, args.uadfv_real_root, "UADFV"
        )
    else:
        print("\n[skip] UADFV: --uadfv_fake_root / --uadfv_real_root not provided.")

    if args.cdfv3_root and args.cdfv3_csv:
        print("\nBuilding CDFv3/CDF++ video list ...")
        dataset_videos["CDFv3"] = build_cdfv3_videos(
            args.cdfv3_root, args.cdfv3_csv
        )
    else:
        print("\n[skip] CDFv3/CDF++: --cdfv3_root / --cdfv3_csv not provided.")

    items = _dataset_balanced_items(dataset_videos, args.max_videos_per_dataset)
    if not items:
        raise ValueError("No dataset items found. Provide at least one dataset.")
    return items


class CombinedVideoDataset(Dataset):
    def __init__(self, items, num_frames: int):
        self.items = items
        self.num_frames = num_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, paths, label, dset = self.items[idx]
        indices = _sample_frame_indices(len(paths), self.num_frames)
        frames = _load_frames([paths[i] for i in indices])
        return frames, label, vid, dset


def load_or_extract_embeddings(args):
    if args.embeddings_npz:
        data = np.load(args.embeddings_npz, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"].astype(int)
        probs = data["probs"] if "probs" in data else np.full(len(labels), np.nan)
        video_ids = data["video_ids"].astype(str).tolist()
        dataset_tags = data["dataset_tags"].astype(str).tolist()
        print(f"Loaded cached embeddings: {embeddings.shape} from {args.embeddings_npz}")
        return embeddings, labels, probs, video_ids, dataset_tags

    if not args.checkpoint:
        raise ValueError("Supply --checkpoint or --embeddings_npz.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, use_memory_bank = load_model(args.checkpoint, args.num_frames, device)
    if use_memory_bank:
        if not args.train_real_root:
            raise ValueError(
                "Checkpoint uses a memory bank; please supply --train_real_root."
            )
        bank = build_memory_bank(
            model,
            args.train_real_root,
            args.num_frames,
            args.knn_k,
            args.bank_batch_size,
            args.num_workers,
            device,
        )
        model.attach_memory_bank(bank)

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    items = build_video_items(args)
    print(f"\nTotal balanced videos: {len(items)}")

    dataset = CombinedVideoDataset(items, args.num_frames)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
    )

    print("\nExtracting video embeddings ...")
    return extract_embeddings(model, loader, device)


def group_indices(labels: np.ndarray, dataset_tags: List[str]):
    tags = np.asarray(dataset_tags)
    groups = []
    for dset in sorted(set(tags.tolist())):
        for label in [0, 1]:
            mask = (tags == dset) & (labels == label)
            if mask.any():
                groups.append((dset, label, np.where(mask)[0]))
    return groups


def compute_similarity_matrices(embeddings: np.ndarray, groups):
    x_norm = l2_normalize(embeddings.astype(np.float64))

    prototypes = []
    group_names = []
    long_group_names = []
    counts = []
    for dset, label, idx in groups:
        proto = l2_normalize(x_norm[idx].mean(axis=0, keepdims=True))[0]
        prototypes.append(proto)
        group_names.append(f"{dset}-{SHORT_LABEL_NAMES[label]}")
        long_group_names.append(f"{dset} {LABEL_NAMES[label]}")
        counts.append(len(idx))

    prototypes = np.stack(prototypes, axis=0)
    proto_sim = prototypes @ prototypes.T

    pairwise_sim = np.zeros_like(proto_sim)
    for i, (_, _, idx_i) in enumerate(groups):
        for j, (_, _, idx_j) in enumerate(groups):
            sim = x_norm[idx_i] @ x_norm[idx_j].T
            if i == j and sim.shape[0] > 1:
                keep = ~np.eye(sim.shape[0], dtype=bool)
                pairwise_sim[i, j] = sim[keep].mean()
            else:
                pairwise_sim[i, j] = sim.mean()

    return proto_sim, pairwise_sim, group_names, long_group_names, counts


def save_matrix_csv(matrix, names, out_path):
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(out_path)


def plot_heatmap(matrix, names, title, out_path, vmin, vmax):
    import matplotlib.pyplot as plt

    size = max(7, 0.75 * len(names) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(groups, pairwise_sim):
    records = []
    for i, (dset_i, label_i, _) in enumerate(groups):
        for j, (dset_j, label_j, _) in enumerate(groups):
            if i >= j or dset_i == dset_j:
                continue
            if label_i == 0 and label_j == 0:
                bucket = "cross_dataset_real_real"
            elif label_i == 1 and label_j == 1:
                bucket = "cross_dataset_fake_fake"
            else:
                bucket = "cross_dataset_real_fake"
            records.append((bucket, pairwise_sim[i, j]))

    print("\nCross-dataset pairwise similarity summary:")
    for bucket in [
        "cross_dataset_real_real",
        "cross_dataset_fake_fake",
        "cross_dataset_real_fake",
    ]:
        vals = [v for b, v in records if b == bucket]
        if vals:
            print(f"  {bucket}: mean={np.mean(vals):.4f}  n_pairs={len(vals)}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, probs, video_ids, dataset_tags = load_or_extract_embeddings(args)
    groups = group_indices(labels, dataset_tags)
    if len(groups) < 2:
        raise ValueError("Need at least two non-empty dataset/label groups.")

    proto_sim, pairwise_sim, names, long_names, counts = compute_similarity_matrices(
        embeddings, groups
    )

    counts_path = out_dir / f"{args.prefix}_group_counts.csv"
    pd.DataFrame(
        {"group": long_names, "short_name": names, "num_videos": counts}
    ).to_csv(counts_path, index=False)

    proto_csv = out_dir / f"{args.prefix}_prototype_similarity.csv"
    pairwise_csv = out_dir / f"{args.prefix}_pairwise_similarity.csv"
    save_matrix_csv(proto_sim, names, proto_csv)
    save_matrix_csv(pairwise_sim, names, pairwise_csv)

    proto_png = out_dir / f"{args.prefix}_prototype_similarity.png"
    pairwise_png = out_dir / f"{args.prefix}_pairwise_similarity.png"
    plot_heatmap(
        proto_sim,
        names,
        "Dataset-label prototype cosine similarity",
        proto_png,
        args.vmin,
        args.vmax,
    )
    plot_heatmap(
        pairwise_sim,
        names,
        "Mean pairwise video cosine similarity",
        pairwise_png,
        args.vmin,
        args.vmax,
    )

    summarize(groups, pairwise_sim)
    print("\nSaved:")
    print(f"  {counts_path}")
    print(f"  {proto_csv}")
    print(f"  {pairwise_csv}")
    print(f"  {proto_png}")
    print(f"  {pairwise_png}")


if __name__ == "__main__":
    main()
