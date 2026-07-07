"""
Visualize DINOv3 embeddings across a scale pyramid on WDF images.

For each sampled WDF frame image:
  1. resize/crop to a base square input,
  2. create scale levels by downsampling to smaller sizes,
  3. feed each level to DINOv3 at its actual smaller resolution,
  4. plot same-image cross-scale similarities,
  5. plot a global UMAP with one panel per scale/resolution.

Example
-------
python visualize_dinov3_scale_pyramid_wdf.py \
  --fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
  --real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
  --num_images 50 \
  --output_dir dinov3_scale_pyramid_wdf \
  --input_size 256 \
  --sizes 256,224,192,160,128,96,64,48 \
  --batch_size 16
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from tqdm import tqdm


RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
WDF_FRAME_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv3 scale-pyramid similarity/UMAP visualization on WDF."
    )
    parser.add_argument(
        "--fake_root",
        default="/media/tarun/B482367C823642E2/usr/wdf/test/fake",
    )
    parser.add_argument(
        "--real_root",
        default="/media/tarun/B482367C823642E2/usr/wdf/test/real",
    )
    parser.add_argument("--num_images", default=50, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--sample_mode", default="balanced", choices=["balanced", "random"])
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument(
        "--scales",
        default="1.0,0.85,0.72,0.60,0.50,0.42,0.35,0.30",
        help="Scale factors. Level size is rounded to a multiple of 16 before being fed.",
    )
    parser.add_argument(
        "--sizes",
        default="256,224,192,160,128,96,64,48",
        help="Comma-separated exact square sizes to feed, e.g. 256,224,192,160,128,96,64,48. Must be divisible by 16. Overrides --scales.",
    )
    parser.add_argument("--output_dir", default="dinov3_scale_pyramid_wdf")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_center_crop", action="store_true")
    parser.add_argument(
        "--model_name",
        default="vit_large_patch16_dinov3.lvd1689m",
        help="timm DINOv3 model name.",
    )
    parser.add_argument(
        "--save_pyramids",
        default=8,
        type=int,
        help="Number of sampled image pyramids to show in a grid.",
    )
    return parser.parse_args()


def parse_scales(text: str) -> list[float]:
    scales = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(scales) < 2:
        raise ValueError("Need at least two scales.")
    if any(s <= 0 or s > 1.0 for s in scales):
        raise ValueError("Scales must be in (0, 1].")
    return scales


def parse_sizes(text: str, input_size: int, patch_size: int = 16) -> list[int]:
    sizes = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(sizes) < 2:
        raise ValueError("Need at least two sizes.")
    if any(s < 2 for s in sizes):
        raise ValueError("Sizes must be >= 2 pixels.")
    if any(s > input_size for s in sizes):
        raise ValueError("Sizes cannot exceed --input_size.")
    if any(s % patch_size != 0 for s in sizes):
        bad = [s for s in sizes if s % patch_size != 0]
        raise ValueError(f"All fed sizes must be divisible by {patch_size}. Bad sizes: {bad}")
    return sizes


def round_to_patch_multiple(size: float, input_size: int, patch_size: int = 16) -> int:
    rounded = int(round(size / patch_size) * patch_size)
    rounded = max(patch_size, rounded)
    return min(input_size, rounded)


def unique_preserve_order(values: list[int]) -> list[int]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def video_id_from_sample_dir(sample_dir: str) -> str:
    sample_dir = str(sample_dir).replace("\\", "/")
    parts = Path(sample_dir).parts
    basename = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = basename.rfind("_frame_")
    if idx != -1:
        video_name = basename[:idx]
    else:
        match = re.search(r"_f\d+$", basename)
        video_name = basename[:match.start()] if match else basename
    return f"{prefix}/{video_name}" if prefix else video_name


def load_wdf_items(fake_root: str, real_root: str) -> list[dict]:
    items = []
    for root_str, label in [(fake_root, 1), (real_root, 0)]:
        root = Path(root_str)
        if not root.is_dir():
            raise FileNotFoundError(f"Root does not exist: {root}")
        found = 0
        skipped = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                skipped += 1
                continue
            match = WDF_FRAME_RE.match(path.name)
            if not match:
                skipped += 1
                continue
            video_id = match.group(1)
            items.append({
                "image_path": str(path),
                "sample_dir": path.name,
                "video_id": video_id,
                "label": label,
            })
            found += 1
        label_name = "fake" if label == 1 else "real"
        print(f"Loaded WDF {label_name}: found={found} skipped={skipped} root={root}")
    return items


def sample_items(items: list[dict], num_images: int, mode: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if mode == "random":
        items = items[:]
        rng.shuffle(items)
        return items[:num_images]

    by_label = {0: [], 1: []}
    for item in items:
        if item["label"] in by_label:
            by_label[item["label"]].append(item)
    for label_items in by_label.values():
        rng.shuffle(label_items)

    half = num_images // 2
    selected = by_label[0][:half] + by_label[1][:num_images - half]
    if len(selected) < num_images:
        remaining = [x for x in items if x not in selected]
        rng.shuffle(remaining)
        selected += remaining[:num_images - len(selected)]
    rng.shuffle(selected)
    return selected


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def make_pyramid_pil(
    image_path: str,
    level_sizes: list[int],
    input_size: int,
    center_crop: bool,
) -> list[Image.Image]:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if center_crop:
        img = center_crop_square(img)
    img = img.resize((input_size, input_size), RESAMPLE_BICUBIC)

    levels = []
    for size in level_sizes:
        if size == input_size:
            level = img.copy()
        else:
            level = img.resize((size, size), RESAMPLE_BICUBIC)
        levels.append(level)
    return levels


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_dinov3(model_name: str, device: torch.device):
    import timm

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.to(device).eval()
    return model


def extract_cls_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    """Return DINOv3 CLS embeddings. Prefer timm forward_intermediates when available."""
    try:
        _, intermediates = model.forward_intermediates(
            images,
            indices=[-1],
            return_prefix_tokens=True,
            norm=True,
        )
        _, prefix_tokens = intermediates[0]
        return prefix_tokens[:, 0, :].float()
    except Exception:
        out = model.forward_features(images)
        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooled", "features"):
                if key in out and out[key] is not None:
                    val = out[key]
                    return val[:, 0, :].float() if val.dim() == 3 else val.float()
            if "x" in out:
                val = out["x"]
                return val[:, 0, :].float() if val.dim() == 3 else val.float()
            raise RuntimeError(f"Unknown forward_features dict keys: {list(out.keys())}")
        if out.dim() == 3:
            return out[:, 0, :].float()
        return out.flatten(1).float()


def batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def save_pyramid_grid(sampled: list[dict], pyramid_images: list[list[Image.Image]], scales, out_path: Path, max_rows: int):
    rows = min(max_rows, len(sampled))
    cols = len(scales)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.65), squeeze=False)
    for r in range(rows):
        label_name = "real" if sampled[r]["label"] == 0 else "fake"
        for c, scale in enumerate(scales):
            ax = axes[r, c]
            ax.imshow(pyramid_images[r][c].resize((128, 128), RESAMPLE_BICUBIC))
            ax.axis("off")
            if r == 0:
                size = pyramid_images[r][c].size[0]
                ax.set_title(f"{scale:g}\n{size}px", fontsize=8)
            if c == 0:
                ax.text(
                    -0.02,
                    0.5,
                    f"{r:02d} {label_name}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
    fig.suptitle("Scale pyramid examples: displayed at common size, but fed to DINOv3 at shown resolution", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_curves(sim_to_full: np.ndarray, labels: np.ndarray, scales, out_path: Path):
    x = np.arange(len(scales))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {0: "#0057FF", 1: "#E31A1C"}
    names = {0: "real", 1: "fake"}
    for label in [0, 1]:
        mask = labels == label
        if not mask.any():
            continue
        for y in sim_to_full[mask]:
            ax.plot(x, y, color=colors[label], alpha=0.12, linewidth=0.8)
        mean = sim_to_full[mask].mean(axis=0)
        std = sim_to_full[mask].std(axis=0)
        ax.plot(x, mean, color=colors[label], linewidth=2.5, label=f"{names[label]} mean")
        ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.14)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in scales])
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Cosine similarity to full-scale embedding")
    ax.set_title("Same-image DINOv3 embedding similarity across scale levels")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adjacent_similarity(adjacent: np.ndarray, labels: np.ndarray, scales, out_path: Path):
    x = np.arange(len(scales) - 1)
    pair_labels = [f"{scales[i]:g}->{scales[i+1]:g}" for i in range(len(scales) - 1)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = {0: "#0057FF", 1: "#E31A1C"}
    names = {0: "real", 1: "fake"}
    for label in [0, 1]:
        mask = labels == label
        if not mask.any():
            continue
        mean = adjacent[mask].mean(axis=0)
        std = adjacent[mask].std(axis=0)
        ax.plot(x, mean, marker="o", color=colors[label], linewidth=2.3, label=names[label])
        ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.14)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30, ha="right")
    ax.set_xlabel("Adjacent scale pair")
    ax.set_ylabel("Adjacent embedding cosine similarity")
    ax.set_title("DINOv3 similarity between neighboring scale levels")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_heatmap(scale_pair_sims: np.ndarray, scales, out_path: Path):
    mean_mat = scale_pair_sims.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mean_mat, vmin=np.percentile(mean_mat, 2), vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(scales)))
    ax.set_yticks(np.arange(len(scales)))
    ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, ha="right")
    ax.set_yticklabels([f"{s:g}" for s in scales])
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Scale factor")
    ax.set_title("Mean same-image cross-scale cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_umap_or_fallback(embeddings_2d_input: np.ndarray, seed: int):
    os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.18,
            metric="cosine",
            random_state=seed,
        )
        coords = reducer.fit_transform(embeddings_2d_input)
        return coords, "UMAP"
    except Exception as exc:
        print(f"[WARNING] UMAP unavailable ({exc}). Falling back to PCA for visualization.")
        coords = PCA(n_components=2, random_state=seed).fit_transform(embeddings_2d_input)
        return coords, "PCA fallback"


def plot_umap_by_scale(coords: np.ndarray, labels: np.ndarray, scales, out_path: Path, method_name: str):
    n_scales = len(scales)
    n_cols = 4
    n_rows = math.ceil(n_scales / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.7), squeeze=False)
    colors = np.where(labels == 0, "#0057FF", "#E31A1C")
    label_names = {0: "real", 1: "fake"}

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    for s_idx, scale in enumerate(scales):
        ax = axes[s_idx // n_cols, s_idx % n_cols]
        mask = np.arange(coords.shape[0]) % n_scales == s_idx
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[mask], s=18, alpha=0.82, linewidths=0)
        ax.set_title(f"scale {scale:g}", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_scales, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label_names[0], markerfacecolor="#0057FF", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label=label_names[1], markerfacecolor="#E31A1C", markersize=6),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle(f"{method_name} of DINOv3 embeddings, shown separately for each scale", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap_by_color_scale(coords: np.ndarray, scale_indices: np.ndarray, scales, out_path: Path, method_name: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scale_indices, cmap="viridis", s=17, alpha=0.78, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_ticks(np.arange(len(scales)))
    cbar.set_ticklabels([f"{s:g}" for s in scales])
    cbar.set_label("Scale factor")
    ax.set_title(f"{method_name} of all image-scale DINOv3 embeddings")
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.input_size % 16 != 0:
        raise ValueError("--input_size must be divisible by 16 for the DINOv3 patch16 backbone.")

    if args.sizes.strip():
        level_sizes = parse_sizes(args.sizes, args.input_size, patch_size=16)
        scales = [size / args.input_size for size in level_sizes]
    else:
        scales = parse_scales(args.scales)
        level_sizes = unique_preserve_order([
            round_to_patch_multiple(args.input_size * scale, args.input_size, patch_size=16)
            for scale in scales
        ])
        scales = [size / args.input_size for size in level_sizes]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("=" * 88)
    print("DINOv3 scale-pyramid visualization on WDF")
    print("=" * 88)
    print(f"Device       : {device}")
    print(f"Model        : {args.model_name}")
    print(f"Fake root    : {args.fake_root}")
    print(f"Real root    : {args.real_root}")
    print(f"Num images   : {args.num_images}")
    print(f"Base size    : {args.input_size}x{args.input_size}")
    print(f"Scales       : {', '.join(f'{s:g}' for s in scales)}")
    print(f"Fed sizes    : {', '.join(f'{s}x{s}' for s in level_sizes)}")
    print(f"Output dir   : {out_dir}")
    print(f"Center crop  : {not args.no_center_crop}")

    items = load_wdf_items(args.fake_root, args.real_root)
    sampled = sample_items(items, args.num_images, args.sample_mode, args.seed)
    labels = np.array([x["label"] for x in sampled], dtype=int)
    print(
        f"Sampled {len(sampled)} images: real={(labels == 0).sum()} "
        f"fake={(labels == 1).sum()}"
    )

    print("\nBuilding image pyramids ...")
    pyramid_images = [
        make_pyramid_pil(
            item["image_path"],
            level_sizes=level_sizes,
            input_size=args.input_size,
            center_crop=not args.no_center_crop,
        )
        for item in tqdm(sampled, desc="Pyramids", unit="image")
    ]
    save_pyramid_grid(
        sampled,
        pyramid_images,
        scales,
        out_dir / "pyramid_examples.png",
        max_rows=args.save_pyramids,
    )

    print("\nLoading DINOv3 ...")
    model = load_dinov3(args.model_name, device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if device.type == "cuda" and not args.fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    print("\nExtracting embeddings ...")
    embeddings_by_scale = []
    n_images = len(sampled)
    n_scales = len(scales)
    with torch.inference_mode(), autocast_ctx:
        for s_idx, scale in enumerate(tqdm(scales, desc="Scales", unit="scale")):
            scale_tensors = torch.stack(
                [pil_to_tensor(pyramid_images[i][s_idx]) for i in range(n_images)],
                dim=0,
            )
            scale_embeddings = []
            for start, end in batched_indices(scale_tensors.shape[0], args.batch_size):
                batch = scale_tensors[start:end].to(device, non_blocking=True)
                scale_embeddings.append(extract_cls_embeddings(model, batch).cpu())
            embeddings_by_scale.append(torch.cat(scale_embeddings, dim=0))

    emb = torch.stack(embeddings_by_scale, dim=1)  # (N images, S scales, C)
    embeddings = emb.reshape(n_images * n_scales, -1)
    embeddings_np = embeddings.numpy().astype(np.float32)
    emb_norm = F.normalize(emb.float(), dim=-1)

    sim_to_full = (emb_norm * emb_norm[:, :1, :]).sum(dim=-1).numpy()
    adjacent = (emb_norm[:, :-1, :] * emb_norm[:, 1:, :]).sum(dim=-1).numpy()
    scale_pair_sims = torch.einsum("isc,itc->ist", emb_norm, emb_norm).numpy()

    plot_similarity_curves(sim_to_full, labels, scales, out_dir / "similarity_to_full_scale.png")
    plot_adjacent_similarity(adjacent, labels, scales, out_dir / "adjacent_scale_similarity.png")
    plot_average_heatmap(scale_pair_sims, scales, out_dir / "mean_cross_scale_similarity_heatmap.png")

    print("\nComputing UMAP/PCA coordinates ...")
    coords, method_name = compute_umap_or_fallback(embeddings_np, args.seed)
    scale_indices = np.tile(np.arange(n_scales), n_images)
    point_labels = np.repeat(labels, n_scales)
    plot_umap_by_scale(coords, point_labels, scales, out_dir / "umap_each_scale.png", method_name)
    plot_umap_by_color_scale(coords, scale_indices, scales, out_dir / "umap_colored_by_scale.png", method_name)

    rows = []
    for i, item in enumerate(sampled):
        for s_idx, scale in enumerate(scales):
            rows.append({
                "image_index": i,
                "sample_dir": item["sample_dir"],
                "image_path": item["image_path"],
                "video_id": item["video_id"],
                "label": item["label"],
                "scale": scale,
                "fed_input_size": pyramid_images[i][s_idx].size[0],
                "sim_to_full": float(sim_to_full[i, s_idx]),
                "umap_x": float(coords[i * n_scales + s_idx, 0]),
                "umap_y": float(coords[i * n_scales + s_idx, 1]),
                "projection": method_name,
            })
    pd.DataFrame(rows).to_csv(out_dir / "scale_embedding_summary.csv", index=False)

    print("\nSaved outputs:")
    for name in [
        "pyramid_examples.png",
        "similarity_to_full_scale.png",
        "adjacent_scale_similarity.png",
        "mean_cross_scale_similarity_heatmap.png",
        "umap_each_scale.png",
        "umap_colored_by_scale.png",
        "scale_embedding_summary.csv",
    ]:
        print(f"  {out_dir / name}")
    print("=" * 88)


if __name__ == "__main__":
    main()
