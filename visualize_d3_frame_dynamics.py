"""
D3-style first- and second-order frame dynamics from an ordered video/frame sequence.

Examples
--------
# One frame folder. Frames are sorted by frame number parsed from filenames/paths.
python visualize_d3_frame_dynamics.py \
  --frames_dir /path/to/video_frames \
  --label "video" \
  --out_dir d3_frame_dynamics_video

# CDF/FF++ style manifest, using one exact video id.
python visualize_d3_frame_dynamics.py \
  --manifest /media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv \
  --root_dir /media/tarun/B482367C823642E2/usr/cdfv1_onct_out \
  --video_id fake/vid_name \
  --label "fake" \
  --out_dir d3_frame_dynamics_cdfv1

# Overlay two videos, similar to the D3 paper figure.
python visualize_d3_frame_dynamics.py \
  --frames_dir /path/to/real_video_frames \
  --label "Real Video" \
  --compare_frames_dir /path/to/fake_video_frames \
  --compare_label "AI-generated Video" \
  --out_dir d3_frame_dynamics_compare
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm


RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot D3-style first/second-order DINOv3 frame dynamics."
    )
    parser.add_argument("--frames_dir", default="", help="Folder containing one video's frames.")
    parser.add_argument("--manifest", default="", help="Optional CSV with sample_dir,label columns.")
    parser.add_argument("--root_dir", default="", help="Root used with --manifest.")
    parser.add_argument("--video_id", default="", help="Video id to select from manifest.")
    parser.add_argument("--label", default="Video")

    parser.add_argument("--compare_frames_dir", default="", help="Optional second frame folder to overlay.")
    parser.add_argument("--compare_manifest", default="", help="Optional second manifest.")
    parser.add_argument("--compare_root_dir", default="", help="Optional second manifest root.")
    parser.add_argument("--compare_video_id", default="", help="Optional second video id from compare manifest.")
    parser.add_argument("--compare_label", default="Compare Video")

    parser.add_argument("--out_dir", default="d3_frame_dynamics")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_frames", default=0, type=int, help="0 means use all frames.")
    parser.add_argument("--delta_t", default=1.0, type=float, help="Frame sampling interval; usually 1.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--model_name",
        default="vit_large_patch16_dinov3.lvd1689m",
        help="timm DINOv3 model name.",
    )
    return parser.parse_args()


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


def frame_index_from_path(path: Path) -> int:
    text = str(path).replace("\\", "/")
    patterns = [
        r"_frame_(\d+)",
        r"_f(\d+)(?:/|\.|$)",
        r"(?:^|/)(\d+)/(?:image\.(?:png|jpg|jpeg|bmp|webp))$",
        r"_(\d+)\.(?:png|jpg|jpeg|bmp|webp)$",
        r"(\d+)(?!.*\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 10**12


def sort_frame_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (frame_index_from_path(p), str(p)))


def collect_from_frames_dir(frames_dir: str) -> list[Path]:
    root = Path(frames_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"frames_dir does not exist: {root}")

    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {root}")
    return sort_frame_paths(image_paths)


def collect_from_manifest(manifest: str, root_dir: str, video_id: str) -> list[Path]:
    if not manifest or not root_dir or not video_id:
        raise ValueError("--manifest, --root_dir, and --video_id are required together.")
    df = pd.read_csv(manifest, sep=None, engine="python")
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found {list(df.columns)}")

    df["video_id"] = df["sample_dir"].apply(video_id_from_sample_dir)
    selected = df[df["video_id"] == video_id].copy()
    if selected.empty:
        available = sorted(df["video_id"].drop_duplicates().astype(str).head(20).tolist())
        raise ValueError(
            f"video_id not found: {video_id}. First available ids include: {available}"
        )

    root = Path(root_dir)
    paths = []
    skipped = 0
    for rel in selected["sample_dir"].astype(str).str.replace("\\", "/", regex=False):
        path = root / rel / "image.png"
        if path.is_file():
            paths.append(path)
        else:
            skipped += 1
    if skipped:
        print(f"  [manifest] skipped {skipped} missing image.png entries")
    if not paths:
        raise FileNotFoundError(f"No existing frames found for video_id={video_id}")
    return sort_frame_paths(paths)


def uniform_subsample(paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    indices = np.unique(indices)
    return [paths[i] for i in indices]


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_frame_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = center_crop_square(img)
    img = img.resize((image_size, image_size), RESAMPLE_BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


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


def embed_frames(paths: list[Path], model, device: torch.device, args) -> torch.Tensor:
    tensors = [load_frame_tensor(path, args.image_size) for path in tqdm(paths, desc="Loading frames")]
    frames = torch.stack(tensors, dim=0)
    embeddings = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if device.type == "cuda" and not args.fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with torch.inference_mode(), autocast_ctx:
        for start, end in tqdm(
            list(batched_indices(len(paths), args.batch_size)),
            desc="Embedding frames",
            unit="batch",
        ):
            batch = frames[start:end].to(device, non_blocking=True)
            embeddings.append(extract_cls_embeddings(model, batch).cpu())
    return torch.cat(embeddings, dim=0)


def compute_d3_features(embeddings: torch.Tensor, delta_t: float) -> dict[str, np.ndarray]:
    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 ordered frames for first- and second-order features.")

    emb = embeddings.float()
    emb_norm = F.normalize(emb, dim=-1)
    l2_first = torch.linalg.vector_norm(emb[1:] - emb[:-1], dim=-1) / delta_t
    cos_first = (emb_norm[1:] * emb_norm[:-1]).sum(dim=-1) / delta_t
    cosdist_first = (1.0 - (emb_norm[1:] * emb_norm[:-1]).sum(dim=-1)) / delta_t

    l2_second_signed = l2_first[1:] - l2_first[:-1]
    cos_second_signed = cos_first[1:] - cos_first[:-1]
    cosdist_second_signed = cosdist_first[1:] - cosdist_first[:-1]

    return {
        "l2_first": l2_first.numpy(),
        "cos_first": cos_first.numpy(),
        "cosdist_first": cosdist_first.numpy(),
        "l2_second": l2_second_signed.abs().numpy(),
        "cos_second": cos_second_signed.abs().numpy(),
        "cosdist_second": cosdist_second_signed.abs().numpy(),
        "l2_second_signed": l2_second_signed.numpy(),
        "cos_second_signed": cos_second_signed.numpy(),
        "cosdist_second_signed": cosdist_second_signed.numpy(),
    }


def collect_sequence(args, prefix: str) -> tuple[list[Path], str] | None:
    if prefix == "":
        frames_dir = args.frames_dir
        manifest = args.manifest
        root_dir = args.root_dir
        video_id = args.video_id
        label = args.label
    else:
        frames_dir = args.compare_frames_dir
        manifest = args.compare_manifest
        root_dir = args.compare_root_dir
        video_id = args.compare_video_id
        label = args.compare_label

    if frames_dir:
        paths = collect_from_frames_dir(frames_dir)
    elif manifest or root_dir or video_id:
        paths = collect_from_manifest(manifest, root_dir, video_id)
    else:
        return None

    paths = uniform_subsample(paths, args.max_frames)
    if len(paths) < 3:
        raise ValueError(f"{label}: need at least 3 frames, found {len(paths)}")
    return paths, label


def plot_d3_figure(series: list[dict], metric: str, out_path: Path):
    if metric == "l2":
        first_key, second_key = "l2_first", "l2_second"
        y1, y2 = "First-order L2 feature", "Second-order L2 feature"
        title = "D3-style L2 frame dynamics"
    elif metric == "cos":
        first_key, second_key = "cos_first", "cos_second"
        y1, y2 = "First-order cosine feature", "Second-order cosine feature"
        title = "D3-style cosine-similarity frame dynamics"
    else:
        first_key, second_key = "cosdist_first", "cosdist_second"
        y1, y2 = "First-order cosine-distance feature", "Second-order cosine-distance feature"
        title = "D3-style cosine-distance frame dynamics"

    colors = ["#0057FF", "#E31A1C", "#111111", "#FFB000"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    for idx, item in enumerate(series):
        color = colors[idx % len(colors)]
        features = item["features"]
        label = item["label"]
        x1 = np.arange(1, len(features[first_key]) + 1)
        axes[0].plot(x1, features[first_key], marker="o", markersize=3.5, linewidth=1.7, color=color, label=label)

        x2 = np.arange(2, len(features[second_key]) + 2)
        if len(series) == 1:
            axes[1].bar(x2, features[second_key], color=color, alpha=0.85, label=label)
        else:
            width = 0.35
            offset = (idx - (len(series) - 1) / 2.0) * width
            axes[1].bar(x2 + offset, features[second_key], width=width, color=color, alpha=0.85, label=label)

    axes[0].set_xlabel("Frame index k")
    axes[0].set_ylabel(y1)
    axes[0].set_title("First-order features")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Frame index k")
    axes[1].set_ylabel(y2)
    axes[1].set_title("Second-order features")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_csv(series: list[dict], out_path: Path):
    rows = []
    for item in series:
        paths = item["paths"]
        features = item["features"]
        label = item["label"]
        for i, path in enumerate(paths):
            row = {
                "series": label,
                "frame_position": i,
                "frame_index": frame_index_from_path(path),
                "path": str(path),
            }
            if i < len(features["l2_first"]):
                row.update({
                    "l2_first": float(features["l2_first"][i]),
                    "cos_first": float(features["cos_first"][i]),
                    "cosdist_first": float(features["cosdist_first"][i]),
                })
            if i < len(features["l2_second"]):
                row.update({
                    "l2_second": float(features["l2_second"][i]),
                    "cos_second": float(features["cos_second"][i]),
                    "cosdist_second": float(features["cosdist_second"][i]),
                    "l2_second_signed": float(features["l2_second_signed"][i]),
                    "cos_second_signed": float(features["cos_second_signed"][i]),
                    "cosdist_second_signed": float(features["cosdist_second_signed"][i]),
                })
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    first = collect_sequence(args, "")
    if first is None:
        raise ValueError("Provide --frames_dir or --manifest/--root_dir/--video_id.")
    sequences.append(first)
    second = collect_sequence(args, "compare_")
    if second is not None:
        sequences.append(second)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("=" * 88)
    print("D3-style frame dynamics visualization")
    print("=" * 88)
    print(f"Device       : {device}")
    print(f"Model        : {args.model_name}")
    print(f"Image size   : {args.image_size}")
    print(f"Delta t      : {args.delta_t}")
    print(f"Output dir   : {out_dir}")
    for paths, label in sequences:
        print(f"  {label}: {len(paths)} frames")
        print(f"    first: {paths[0]}")
        print(f"    last : {paths[-1]}")
    if len(sequences) > 1 and len({len(paths) for paths, _ in sequences}) > 1:
        print("  [warning] compared videos have different frame counts; use --max_frames for cleaner overlays.")

    print("\nLoading DINOv3 ...")
    model = load_dinov3(args.model_name, device)

    series = []
    for paths, label in sequences:
        print(f"\nExtracting embeddings for {label} ...")
        embeddings = embed_frames(paths, model, device, args)
        features = compute_d3_features(embeddings, delta_t=args.delta_t)
        series.append({
            "label": label,
            "paths": paths,
            "embeddings": embeddings,
            "features": features,
        })
        print(
            f"  {label}: L2 first-order std={features['l2_first'].std():.4f}, "
            f"cosdist first-order std={features['cosdist_first'].std():.4f}"
        )

    plot_d3_figure(series, "l2", out_dir / "d3_frame_dynamics_l2.png")
    plot_d3_figure(series, "cos", out_dir / "d3_frame_dynamics_cosine_similarity.png")
    plot_d3_figure(series, "cosdist", out_dir / "d3_frame_dynamics_cosine_distance.png")
    save_csv(series, out_dir / "d3_frame_dynamics_features.csv")

    print("\nSaved outputs:")
    for name in [
        "d3_frame_dynamics_l2.png",
        "d3_frame_dynamics_cosine_similarity.png",
        "d3_frame_dynamics_cosine_distance.png",
        "d3_frame_dynamics_features.csv",
    ]:
        print(f"  {out_dir / name}")
    print("=" * 88)


if __name__ == "__main__":
    main()
