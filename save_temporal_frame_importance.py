"""
Visualize temporal frame importance from a Stage-2 frame-end checkpoint.

The script samples 8 frames uniformly from a user-provided folder of extracted
frames, runs the Stage-2 model, and visualizes the deepest temporal stream
(DINO block 23):

  1. frame-importance timeline from VID-token attention to frame tokens
  2. full temporal attention matrix over [VID, f1, ..., f8]

Example:
    python save_temporal_frame_importance.py \
        --frame_folder /path/to/video_frames \
        --checkpoint checkpoints_s2_frame_end/best.pth \
        --output_dir temporal_viz
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn

from frame_model import ViT
from video_model import TemporalTransformer


IMG_SIZE = 256
NUM_VIS_FRAMES = 8
STREAM_INDEX = 3
STREAM_NAME = "layer23"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save temporal frame-importance timeline and attention matrix."
    )
    parser.add_argument("--frame_folder", "--folder", required=True, type=str)
    parser.add_argument("--checkpoint", default="checkpoints_s2_frame_end/best.pth", type=str)
    parser.add_argument(
        "--stage1_checkpoint",
        default="",
        type=str,
        help="Optional Stage-1 checkpoint if --checkpoint does not contain frame_model weights.",
    )
    parser.add_argument("--output_dir", default="temporal_frame_importance", type=str)
    parser.add_argument("--num_frames", default=NUM_VIS_FRAMES, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj)))
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def checkpoint_num_frames(state: dict, default: int = 32) -> int:
    key = "temporal_transformers.0.pos_embed"
    if key in state and state[key].ndim == 3:
        return int(state[key].shape[1])
    return default


def checkpoint_fusion_in_dim(state: dict) -> int:
    key = "fusion_classifier.weight"
    if key in state and state[key].ndim == 2:
        return int(state[key].shape[1])
    return ViT.NUM_LAYERS * ViT.EMBED_DIM + 2


class FrameEndVideoViT(nn.Module):
    """Minimal Stage-2 model matching train_stage2_frame_end.py for inference."""

    EMBED_DIM = ViT.EMBED_DIM
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
        fusion_in_dim: int = ViT.NUM_LAYERS * ViT.EMBED_DIM + 2,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim=self.EMBED_DIM,
                num_frames=num_frames,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])
        self.fusion_classifier = nn.Linear(fusion_in_dim, 2)

    @staticmethod
    def _mean_frame_logits(frame_logits_list: list, bsz: int, num_frames: int, dtype: torch.dtype):
        return frame_logits_list[-1].float().reshape(bsz, num_frames, 2).mean(dim=1).to(dtype=dtype)


def load_model(args, device: torch.device) -> FrameEndVideoViT:
    ckpt_path = Path(args.checkpoint).expanduser()
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    has_temporal = any(k.startswith("temporal_transformers.") for k in state)
    if not has_temporal:
        raise ValueError(
            f"{ckpt_path} does not look like a Stage-2 checkpoint: "
            "no temporal_transformers.* keys found."
        )

    model = FrameEndVideoViT(
        num_frames=checkpoint_num_frames(state),
        fusion_in_dim=checkpoint_fusion_in_dim(state),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)

    has_frame_prefix = any(k.startswith("frame_model.") for k in state)
    if not has_frame_prefix:
        if not args.stage1_checkpoint:
            raise ValueError(
                "Stage-2 checkpoint has no frame_model.* weights. "
                "Pass --stage1_checkpoint with the Stage-1 frame model checkpoint."
            )
        load_stage1_weights(model, args.stage1_checkpoint)

    model.to(device).eval()
    model.frame_model.eval()
    print(f"Loaded Stage-2 checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"  first unexpected keys: {unexpected[:5]}")
    return model


def load_stage1_weights(model: FrameEndVideoViT, checkpoint: str):
    ckpt_path = Path(checkpoint).expanduser()
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if any(k.startswith("frame_model.") for k in state):
        state = {k[len("frame_model."):]: v for k, v in state.items() if k.startswith("frame_model.")}
    missing, unexpected = model.frame_model.load_state_dict(state, strict=False)
    print(f"Loaded Stage-1 frame weights: {ckpt_path}")
    print(f"  frame missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def sample_frame_folder(frame_folder: str, num_frames: int):
    folder = Path(frame_folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame folder not found: {folder}")

    all_paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )
    if not all_paths:
        raise RuntimeError(f"No image frames found in: {folder}")
    if len(all_paths) < num_frames:
        raise RuntimeError(
            f"Need at least {num_frames} frames, but found only {len(all_paths)} in {folder}"
        )

    pick = np.linspace(0, len(all_paths) - 1, num_frames)
    pick = np.round(pick).astype(int).tolist()
    sampled_paths = [all_paths[i] for i in pick]

    frames = []
    for path in sampled_paths:
        try:
            frames.append(Image.open(path).convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not read sampled frame {path}: {exc}") from exc

    print(f"Frame folder: {folder}")
    print(f"  discovered image frames: {len(all_paths)}")
    print(f"  sampled sorted positions: {pick}")
    print("  sampled files:")
    for slot, (pos, path) in enumerate(zip(pick, sampled_paths), start=1):
        print(f"    f{slot}: pos={pos:03d}  file={path.name}")

    return frames, pick, sampled_paths


def preprocess_frames(frames: list[Image.Image]) -> Tensor:
    tensors = []
    for image in frames:
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(tensor)
    return torch.cat(tensors, dim=0).unsqueeze(0)


def _self_attn_with_weights(layer: nn.TransformerEncoderLayer, x: Tensor):
    q = layer.norm1(x)
    try:
        return layer.self_attn(
            q, q, q,
            need_weights=True,
            average_attn_weights=False,
            is_causal=False,
        )
    except TypeError:
        try:
            return layer.self_attn(
                q, q, q,
                need_weights=True,
                average_attn_weights=False,
            )
        except TypeError:
            return layer.self_attn(q, q, q, need_weights=True)


def temporal_forward_with_last_attention(
    temporal_tfm: TemporalTransformer,
    frame_cls: Tensor,
) -> tuple[Tensor, Tensor]:
    bsz, num_frames, _ = frame_cls.shape
    x = frame_cls + temporal_tfm.pos_embed[:, :num_frames, :]
    video_token = temporal_tfm.video_token.expand(bsz, -1, -1)
    x = torch.cat([video_token, x], dim=1)

    layers = temporal_tfm.encoder.layers
    for layer in layers[:-1]:
        x = layer(x)

    last = layers[-1]
    if not getattr(last, "norm_first", False):
        raise RuntimeError("Expected norm_first=True in TemporalTransformer.")

    attn_out, attn_weights = _self_attn_with_weights(last, x)
    x = x + last.dropout1(attn_out)
    x = x + last._ff_block(last.norm2(x))
    x = temporal_tfm.norm(x)
    return x[:, 0], attn_weights


@torch.no_grad()
def run_model_and_attention(model: FrameEndVideoViT, video: Tensor, amp_enabled: bool):
    bsz, num_frames, channels, height, width = video.shape
    if bsz != 1:
        raise ValueError("This visualization script expects one video at a time.")
    if num_frames > model.num_frames:
        raise ValueError(f"Model supports <= {model.num_frames} frames, got {num_frames}.")

    frames = video.reshape(bsz * num_frames, channels, height, width)
    autocast_device = "cuda" if video.is_cuda else "cpu"
    with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
        frame_logits_list, _, cls_list = model.frame_model(frames)
        cls_sequences = [
            cls_tokens.reshape(bsz, num_frames, model.EMBED_DIM) for cls_tokens in cls_list
        ]

        video_feats = []
        final_attn = None
        for idx, (temporal_tfm, frame_cls) in enumerate(zip(model.temporal_transformers, cls_sequences)):
            if idx == STREAM_INDEX:
                feat, attn = temporal_forward_with_last_attention(temporal_tfm, frame_cls)
                final_attn = attn.float()
            else:
                feat = temporal_tfm(frame_cls)
            video_feats.append(feat)

        temporal_vec = torch.cat(video_feats, dim=1)
        if model.fusion_classifier.in_features == temporal_vec.shape[1] + 2:
            frame_mean_logits = model._mean_frame_logits(
                frame_logits_list, bsz, num_frames, temporal_vec.dtype
            )
            fused = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        elif model.fusion_classifier.in_features == temporal_vec.shape[1]:
            fused = temporal_vec
        else:
            raise RuntimeError(
                "fusion_classifier input size does not match temporal features "
                f"({model.fusion_classifier.in_features} vs {temporal_vec.shape[1]} or "
                f"{temporal_vec.shape[1] + 2})."
            )
        video_logits = model.fusion_classifier(fused)

    if final_attn is None:
        raise RuntimeError("Failed to capture temporal attention.")

    # Preferred shape: (B, heads, tokens, tokens). Older torch versions may
    # return already-averaged weights: (B, tokens, tokens).
    if final_attn.ndim == 4:
        attn_matrix = final_attn[0].mean(dim=0).cpu().numpy()
    elif final_attn.ndim == 3:
        attn_matrix = final_attn[0].cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected attention weight shape: {tuple(final_attn.shape)}")
    video_prob = torch.softmax(video_logits.float(), dim=1)[0].cpu().numpy()
    frame_probs = torch.softmax(frame_logits_list[-1].float(), dim=1)[:, 1].cpu().numpy()
    return video_prob, frame_probs, attn_matrix


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def heat_color(value: float):
    value = float(np.clip(value, 0.0, 1.0))
    # Blue -> yellow -> red, readable on white backgrounds.
    if value < 0.5:
        t = value / 0.5
        r = int(45 + t * (245 - 45))
        g = int(105 + t * (210 - 105))
        b = int(185 + t * (70 - 185))
    else:
        t = (value - 0.5) / 0.5
        r = int(245 + t * (190 - 245))
        g = int(210 + t * (45 - 210))
        b = int(70 + t * (35 - 70))
    return r, g, b


def get_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:80]


def save_sampled_frames(frames: list[Image.Image], indices: list[int], paths: list[Path], out_dir: Path):
    frame_dir = out_dir / "sampled_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, (image, idx, path) in enumerate(zip(frames, indices, paths), start=1):
        image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC).save(
            frame_dir / f"frame_{i:02d}_pos_{idx:03d}_{safe_stem(path)}.png"
        )


def save_scores_csv(
    out_path: Path,
    indices: list[int],
    paths: list[Path],
    raw_scores: np.ndarray,
    norm_scores: np.ndarray,
    frame_probs: np.ndarray,
):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slot",
            "sorted_position",
            "filename",
            "vid_to_frame_attention",
            "normalized_score",
            "frame_fake_prob",
        ])
        for i, (idx, path, raw, norm, prob) in enumerate(
            zip(indices, paths, raw_scores, norm_scores, frame_probs),
            start=1,
        ):
            writer.writerow([i, idx, path.name, f"{raw:.8f}", f"{norm:.8f}", f"{prob:.8f}"])


def save_timeline(
    frames: list[Image.Image],
    indices: list[int],
    scores: np.ndarray,
    frame_probs: np.ndarray,
    video_prob: np.ndarray,
    out_path: Path,
) -> Image.Image:
    norm_scores = normalize(scores)
    thumb = 142
    gap = 16
    margin = 28
    title_h = 58
    bar_h = 18
    score_h = 44
    width = margin * 2 + len(frames) * thumb + (len(frames) - 1) * gap
    height = title_h + thumb + score_h + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(22, bold=True)
    font = get_font(14)
    small = get_font(12)

    fake_prob = float(video_prob[1])
    pred = "Fake" if fake_prob >= 0.5 else "Real"
    title = f"Temporal Frame-Importance Timeline ({STREAM_NAME})  |  Prediction: {pred}, p(fake)={fake_prob:.3f}"
    draw.text((margin, 18), title, fill=(20, 20, 20), font=title_font)

    y_img = title_h
    for i, (image, idx, raw_score, norm_score, frame_prob) in enumerate(
        zip(frames, indices, scores, norm_scores, frame_probs)
    ):
        x = margin + i * (thumb + gap)
        resized = image.resize((thumb, thumb), Image.BICUBIC)
        canvas.paste(resized, (x, y_img))
        draw.rectangle((x, y_img, x + thumb - 1, y_img + thumb - 1), outline=(40, 40, 40), width=1)

        bar_y = y_img + thumb + 10
        draw.rectangle((x, bar_y, x + thumb, bar_y + bar_h), fill=(230, 230, 230))
        draw.rectangle(
            (x, bar_y, x + int(round(thumb * float(norm_score))), bar_y + bar_h),
            fill=heat_color(float(norm_score)),
        )
        draw.rectangle((x, bar_y, x + thumb, bar_y + bar_h), outline=(80, 80, 80), width=1)

        draw.text((x, bar_y + 24), f"f{i} idx {idx}", fill=(25, 25, 25), font=small)
        draw.text((x, bar_y + 39), f"attn {raw_score:.3f} | fp {frame_prob:.2f}", fill=(25, 25, 25), font=small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")
    return canvas


def save_attention_matrix(attn_matrix: np.ndarray, out_path: Path) -> Image.Image:
    labels = ["VID"] + [f"f{i}" for i in range(1, attn_matrix.shape[0])]
    cell = 64
    left = 76
    top = 72
    right = 28
    bottom = 42
    width = left + cell * len(labels) + right
    height = top + cell * len(labels) + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(22, bold=True)
    font = get_font(14)
    small = get_font(11)

    draw.text((24, 18), f"Temporal Attention Matrix ({STREAM_NAME}, final transformer block)", fill=(20, 20, 20), font=title_font)
    max_val = max(float(attn_matrix.max()), 1e-8)
    for i, label in enumerate(labels):
        x = left + i * cell + cell // 2
        draw.text((x - 14, top - 28), label, fill=(20, 20, 20), font=font)
        y = top + i * cell + cell // 2
        draw.text((22, y - 8), label, fill=(20, 20, 20), font=font)

    for r in range(len(labels)):
        for c in range(len(labels)):
            value = float(attn_matrix[r, c])
            color = heat_color(value / max_val)
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(255, 255, 255))
            text = f"{value:.2f}"
            fill = (255, 255, 255) if value / max_val > 0.58 else (15, 15, 15)
            draw.text((x0 + 17, y0 + 24), text, fill=fill, font=small)

    draw.text((left, height - 26), "Rows attend to columns; heads are averaged.", fill=(50, 50, 50), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")
    return canvas


def save_combined(timeline: Image.Image, matrix: Image.Image, out_path: Path):
    pad = 20
    width = max(timeline.width, matrix.width) + pad * 2
    height = timeline.height + matrix.height + pad * 3
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(timeline, ((width - timeline.width) // 2, pad))
    canvas.paste(matrix, ((width - matrix.width) // 2, timeline.height + pad * 2))
    canvas.save(out_path)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    if args.num_frames != NUM_VIS_FRAMES:
        print(f"Using {args.num_frames} sampled frames; the matrix will be {(args.num_frames + 1)}x{(args.num_frames + 1)}.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = (device.type == "cuda") and not args.no_amp
    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, indices, sampled_paths = sample_frame_folder(args.frame_folder, args.num_frames)
    save_sampled_frames(frames, indices, sampled_paths, out_dir)

    model = load_model(args, device)
    video_tensor = preprocess_frames(frames).to(device)
    video_prob, frame_probs, attn_matrix = run_model_and_attention(model, video_tensor, amp_enabled)

    importance_scores = attn_matrix[0, 1:]
    norm_scores = normalize(importance_scores)
    pred = "Fake" if float(video_prob[1]) >= 0.5 else "Real"

    print("\nPrediction")
    print(f"  predicted label: {pred}")
    print(f"  p(real): {float(video_prob[0]):.6f}")
    print(f"  p(fake): {float(video_prob[1]):.6f}")
    print("\nFrame importance from VID -> frame attention")
    for i, (idx, path, raw, norm, fp) in enumerate(
        zip(indices, sampled_paths, importance_scores, norm_scores, frame_probs),
        start=1,
    ):
        print(
            f"  f{i}: pos={idx:03d}  file={path.name}  "
            f"attn={raw:.6f}  norm={norm:.6f}  frame_p_fake={fp:.6f}"
        )

    save_scores_csv(
        out_dir / "temporal_frame_scores.csv",
        indices,
        sampled_paths,
        importance_scores,
        norm_scores,
        frame_probs,
    )
    timeline = save_timeline(
        frames,
        indices,
        importance_scores,
        frame_probs,
        video_prob,
        out_dir / "timeline.png",
    )
    matrix = save_attention_matrix(attn_matrix, out_dir / "temporal_attention_matrix.png")
    save_combined(timeline, matrix, out_dir / "combined_temporal_viz.png")


if __name__ == "__main__":
    main()
