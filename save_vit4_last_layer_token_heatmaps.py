"""
Visualize MAC-style last-layer attention maps for checkpoints_vit_4layers.

Model:
  frame_model.py::ViT loaded from checkpoints_vit_4layers/best.pth

Input:
  one user-provided image, resized to 256x256.

Outputs for the last tapped layer only, i.e. DINOv3 block 23:
  - CLS: final-block self-attention from CLS query to patch keys
  - REG: final-block self-attention from REG queries to patch keys
  - Patch/AVG: final-block self-attention from patch queries to patch keys
  - Patch*: learned patch attention-pooling map from SpatialHead.patch_pool

CLS, REG, and Patch/AVG follow the DINO-MAC visualization style: maps are
averaged across ViT attention heads and, for REG/Patch, across tokens in the
group. Patch* is shown separately because it is our SpatialHead's learned
patch-pooling attention.
"""

import argparse
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


def make_panel(base_image: Image.Image, grid: np.ndarray, title: str, alpha: float) -> Image.Image:
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
    return canvas


def save_heatmap(base_image: Image.Image, grid: np.ndarray, out_path: Path, title: str, alpha: float):
    canvas = make_panel(base_image, grid, title, alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")


def make_input_panel(base_image: Image.Image, title: str = "Input") -> Image.Image:
    canvas = Image.new("RGB", (base_image.width, base_image.height + 30), "white")
    canvas.paste(base_image.convert("RGB"), (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(0, 0, 0), font=font)
    return canvas


def save_contact_sheet(base_image: Image.Image, maps: dict, out_path: Path, alpha: float):
    panels = [
        make_input_panel(base_image, "Input"),
        make_panel(base_image, maps["cls"], "CLS", alpha),
        make_panel(base_image, maps["reg"], "REG", alpha),
        make_panel(base_image, maps["patch"], "Patch/AVG", alpha),
        make_panel(base_image, maps["patch_star"], "Patch*", alpha),
    ]
    sheet = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Saved {out_path}")


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


def get_blocks(vit_module):
    candidates = [vit_module]
    for attr in ("base_model", "model"):
        obj = getattr(candidates[-1], attr, None)
        if obj is not None:
            candidates.append(obj)
    base_model = getattr(vit_module, "base_model", None)
    if base_model is not None and getattr(base_model, "model", None) is not None:
        candidates.append(base_model.model)

    for obj in candidates:
        blocks = getattr(obj, "blocks", None)
        if blocks is not None:
            return blocks
    raise AttributeError("Could not find ViT blocks on model.vit or its PEFT-wrapped base model.")


def compute_attention_matrix(attn_module, attn_input: torch.Tensor) -> torch.Tensor:
    bsz, num_tokens, _ = attn_input.shape
    qkv = attn_module.qkv(attn_input)
    qkv = qkv.reshape(
        bsz,
        num_tokens,
        3,
        attn_module.num_heads,
        attn_module.head_dim,
    ).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q = attn_module.q_norm(q)
    k = attn_module.k_norm(k)
    attn = (q.float() * attn_module.scale) @ k.float().transpose(-2, -1)
    return attn.softmax(dim=-1)


def self_attention_grids(attn: torch.Tensor, num_reg: int, height: int, width: int) -> dict:
    patch_start = 1 + num_reg
    patch_end = patch_start + height * width
    if attn.shape[-1] < patch_end:
        raise ValueError(
            f"Attention has {attn.shape[-1]} tokens, but expected at least {patch_end} "
            f"for CLS + {num_reg} REG + {height * width} patches."
        )

    patch_cols = slice(patch_start, patch_end)
    cls_map = attn[0, :, 0, patch_cols].mean(dim=0)
    reg_map = attn[0, :, 1:patch_start, patch_cols].mean(dim=(0, 1))
    patch_map = attn[0, :, patch_start:patch_end, patch_cols].mean(dim=(0, 1))

    return {
        "cls": cls_map.detach().cpu().numpy().reshape(height, width),
        "reg": reg_map.detach().cpu().numpy().reshape(height, width),
        "patch": patch_map.detach().cpu().numpy().reshape(height, width),
    }


@torch.no_grad()
def compute_last_layer_maps(model: ViT, image_tensor: torch.Tensor):
    last_layer = model.LAYERS[-1]
    spatial_head = model.spatial_heads[-1]
    blocks = get_blocks(model.vit)
    final_block = blocks[last_layer]
    captured = {}

    def capture_attn_input(_module, inputs):
        captured["attn_input"] = inputs[0].detach().clone()

    handle = final_block.attn.register_forward_pre_hook(capture_attn_input)
    try:
        _, intermediates = model.vit.forward_intermediates(
            image_tensor,
            indices=[last_layer],
            return_prefix_tokens=True,
            norm=True,
        )
    finally:
        handle.remove()

    if "attn_input" not in captured:
        raise RuntimeError("Could not capture the final-block attention input.")

    spatial_map, prefix_tokens = intermediates[0]
    bsz, channels, height, width = spatial_map.shape
    patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
        bsz, height * width, channels
    )

    cls_tok_full = prefix_tokens[:, :1, :]
    reg_tok_full = prefix_tokens[:, 1:1 + model.NUM_REG, :]

    final_attn = compute_attention_matrix(final_block.attn, captured["attn_input"])
    maps = self_attention_grids(final_attn, model.NUM_REG, height, width)

    patch_star_attn = patch_attention_from_pool(spatial_head.patch_pool, patch_tokens)
    patch_star_map = patch_star_attn[0].mean(dim=0).detach().cpu().numpy().reshape(height, width)

    result = spatial_head(cls_tok_full, reg_tok_full, patch_tokens)
    probs = torch.softmax(result["logits"], dim=1)[0].detach().cpu().numpy()

    return {
        "cls": maps["cls"],
        "reg": maps["reg"],
        "patch": maps["patch"],
        "patch_star": patch_star_map,
        "probs": probs,
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

    real_prob = float(maps["probs"][0])
    fake_prob = float(maps["probs"][1])
    pred_label = "fake" if fake_prob >= real_prob else "real"
    pred_prob = max(real_prob, fake_prob)
    print(f"\nPrediction from last-layer head: {pred_label} ({pred_prob:.4f})")
    print(f"  real probability: {real_prob:.4f}")
    print(f"  fake probability: {fake_prob:.4f}")

    save_heatmap(base_image, maps["cls"], out_dir / "01_cls_self_attention.png", "CLS | final block attention", args.alpha)
    save_heatmap(base_image, maps["reg"], out_dir / "02_reg_self_attention.png", "REG | final block attention", args.alpha)
    save_heatmap(base_image, maps["patch"], out_dir / "03_patch_avg_self_attention.png", "Patch/AVG | final block attention", args.alpha)
    save_heatmap(base_image, maps["patch_star"], out_dir / "04_patch_star_pooling_attention.png", "Patch* | pooling attention", args.alpha)
    save_contact_sheet(base_image, maps, out_dir / "00_mac_style_contact_sheet.png", args.alpha)

    print(f"\nDone. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
