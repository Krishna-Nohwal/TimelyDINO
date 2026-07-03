"""
Visualize last-layer token maps for checkpoints_vit_4layers.

Model:
  frame_model.py::ViT loaded from checkpoints_vit_4layers/best.pth

Input:
  one user-provided image, resized to 256x256.

Outputs for the last tapped layer only, i.e. DINOv3 block 23:
  - CLS: cosine-similarity map between CLS token and patch tokens
  - REG: cosine-similarity map between mean register token and patch tokens
  - Patch: uniform patch-average pooling map
  - Patch*: learned patch attention-pooling map from SpatialHead.patch_pool

Patch* means the attention-pooled patch branch. Patch means average pooling.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from frame_model import ViT


IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path: str):
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, image


def clean_state_dict(state):
    state = state.get("state_dict", state.get("model_state_dict", state.get("model", state)))
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_vit4(checkpoint: str, device: torch.device) -> ViT:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ViT()
    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_heatmap(base_image: Image.Image, grid: np.ndarray, out_path: Path, title: str, alpha: float):
    grid = normalize_map(grid)
    heat = jet_colormap(grid)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(base_image.size, Image.BICUBIC)
    overlay = Image.blend(base_image.convert("RGB"), heat_img.convert("RGB"), alpha)

    canvas = Image.new("RGB", (overlay.width, overlay.height + 30), "white")
    canvas.paste(overlay, (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")


def cosine_grid(query: torch.Tensor, patches: torch.Tensor, height: int, width: int) -> np.ndarray:
    query = torch.nn.functional.normalize(query.float(), dim=-1)
    patches = torch.nn.functional.normalize(patches.float(), dim=-1)
    scores = (patches * query.unsqueeze(1)).sum(dim=-1)
    return scores[0].detach().cpu().numpy().reshape(height, width)


def patch_attention_from_pool(pool, patch_tokens: torch.Tensor) -> torch.Tensor:
    patch_tokens = patch_tokens.detach().clone()
    bsz, num_tokens, _ = patch_tokens.shape
    x_heads = patch_tokens.view(
        bsz, num_tokens, pool.num_heads, pool.head_dim
    ).permute(0, 2, 1, 3)
    q = pool.query.expand(bsz, -1, -1, -1)
    attn = (q @ x_heads.transpose(-2, -1)) * pool.scale
    attn = attn + pool._positional_bias(num_tokens, patch_tokens.device, attn.dtype)
    return attn.softmax(dim=-1).squeeze(2)  # (B, pool_heads, N)


@torch.no_grad()
def compute_last_layer_maps(model: ViT, image_tensor: torch.Tensor):
    last_layer = model.LAYERS[-1]
    spatial_head = model.spatial_heads[-1]

    _, intermediates = model.vit.forward_intermediates(
        image_tensor,
        indices=[last_layer],
        return_prefix_tokens=True,
        norm=True,
    )

    spatial_map, prefix_tokens = intermediates[0]
    bsz, channels, height, width = spatial_map.shape
    patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
        bsz, height * width, channels
    )

    cls_tok = prefix_tokens[:, 0, :]
    reg_tok = prefix_tokens[:, 1:1 + model.NUM_REG, :].mean(dim=1)

    cls_map = cosine_grid(cls_tok, patch_tokens, height, width)
    reg_map = cosine_grid(reg_tok, patch_tokens, height, width)
    patch_avg_map = np.ones((height, width), dtype=np.float32)

    patch_star_attn = patch_attention_from_pool(spatial_head.patch_pool, patch_tokens)
    patch_star_map = patch_star_attn[0].mean(dim=0).detach().cpu().numpy().reshape(height, width)

    return {
        "cls": cls_map,
        "reg": reg_map,
        "patch": patch_avg_map,
        "patch_star": patch_star_map,
    }


def main():
    parser = argparse.ArgumentParser("Save last-layer CLS/REG/Patch/Patch* maps for checkpoints_vit_4layers")
    parser.add_argument("--image", required=True, type=str, help="Input image path.")
    parser.add_argument("--checkpoint", default="checkpoints_vit_4layers/best.pth", type=str)
    parser.add_argument("--out_dir", default="vit4_last_layer_token_heatmaps", type=str)
    parser.add_argument("--alpha", default=0.45, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Using last tapped layer only: DINOv3 block 23")
    print("Input image will be resized to 256x256")

    image_tensor, base_image = load_image(args.image)
    image_tensor = image_tensor.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_image.save(out_dir / "input_256.png")

    model = load_vit4(args.checkpoint, device)
    maps = compute_last_layer_maps(model, image_tensor)

    save_heatmap(base_image, maps["cls"], out_dir / "01_cls_heatmap.png", "CLS | layer 23", args.alpha)
    save_heatmap(base_image, maps["reg"], out_dir / "02_reg_heatmap.png", "REG | layer 23", args.alpha)
    save_heatmap(base_image, maps["patch"], out_dir / "03_patch_avg_heatmap.png", "Patch avg | layer 23", args.alpha)
    save_heatmap(base_image, maps["patch_star"], out_dir / "04_patch_star_attention_heatmap.png", "Patch* attention | layer 23", args.alpha)

    print(f"\nDone. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
