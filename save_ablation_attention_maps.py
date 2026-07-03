"""
Save patch attention / patch-importance maps for one or more input images
across the Stage-1 ablation checkpoints.

Actual learned attention maps are saved for:
  - checkpoints_patch_attn
  - checkpoints_last_layer_all
  - checkpoints_vit_4layers

The CLS, REG, and patch-average checkpoints do not contain an attention-pooling
module. For those, this script saves clearly named comparison maps:
  - CLS: cosine similarity between the CLS token and each patch token
  - REG: cosine similarity between mean register token and each patch token
  - Patch average: uniform pooling weights
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from frame_model import ViT as FourLayerViT


IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path: str, size: int = IMG_SIZE) -> Tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, image


def resolve_sample_image(sample_dir: str, root_dir: str) -> str:
    rel = str(sample_dir).replace("\\", "/")
    root = Path(root_dir)
    candidates = []

    rel_path = Path(rel)
    if rel_path.is_absolute():
        candidates.append(rel_path if rel_path.name == "image.png" else rel_path / "image.png")

    candidates.append(root / rel / "image.png")
    if "sampled_30k/" in rel:
        candidates.append(root / rel.split("sampled_30k/", 1)[1] / "image.png")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def sample_images_from_manifest(manifest: str, root_dir: str, num_images: int, seed: int) -> list[str]:
    df = pd.read_csv(manifest)
    if "sample_dir" not in df.columns:
        raise ValueError(f"Manifest must contain sample_dir. Found: {list(df.columns)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    selected = []
    missing = 0
    for idx in order:
        path = resolve_sample_image(df.iloc[int(idx)]["sample_dir"], root_dir)
        if os.path.isfile(path):
            selected.append(path)
            if len(selected) >= num_images:
                break
        else:
            missing += 1

    if len(selected) < num_images:
        raise RuntimeError(
            f"Only found {len(selected)} valid images from {manifest}; "
            f"missing checked entries={missing}."
        )
    return selected


def safe_sample_name(path: str, idx: int) -> str:
    p = Path(path)
    parent = p.parent.name or "image"
    grandparent = p.parent.parent.name or "sample"
    name = f"{idx:03d}_{grandparent}_{parent}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def make_backbone(drop_path: float = 0.10):
    vit = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        drop_path_rate=drop_path,
    )
    vit = get_peft_model(vit, LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["attn.qkv"],
        lora_dropout=0.10,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    ))
    vit.base_model.model.set_grad_checkpointing(enable=True)
    return vit


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 4, num_patches: int = 256):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_patches = num_patches
        self.scale = self.head_dim ** -0.5
        self.query = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, 1, num_patches))

    def _positional_bias(self, n: int, device, dtype):
        bias = self.pos_bias
        if n != self.num_patches:
            bias = nn.functional.interpolate(
                bias.reshape(1, 1, self.num_patches),
                size=n,
                mode="linear",
                align_corners=False,
            ).reshape(1, 1, 1, n)
        return bias.to(device=device, dtype=dtype)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        x_heads = x.view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.query.expand(bsz, -1, -1, -1)
        attn = (q @ x_heads.transpose(-2, -1)) * self.scale
        attn = attn + self._positional_bias(num_tokens, x.device, attn.dtype)
        return attn.softmax(dim=-1).squeeze(2)  # (B, heads, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, channels = x.shape
        x_heads = x.view(bsz, x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = self.attention(x).unsqueeze(2)
        out = attn @ x_heads
        return out.permute(0, 2, 1, 3).reshape(bsz, channels)


class BaseLastLayerModel(nn.Module):
    EMBED_DIM = 1024
    NUM_REG = 4
    FINAL_LAYER = 23
    DROP_PATH = 0.10

    def final_outputs(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        spatial_map, prefix_tokens = intermediates[0]
        bsz, channels, height, width = spatial_map.shape
        patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
            bsz, height * width, channels
        )
        return spatial_map, prefix_tokens, patch_tokens, height, width


class CLSViT(BaseLastLayerModel):
    def __init__(self):
        super().__init__()
        self.vit = make_backbone(self.DROP_PATH)
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, prefix_tokens, _, _, _ = self.final_outputs(x)
        feat = prefix_tokens[:, 0, :]
        return self.classifier(feat.float()), feat


class RegViT(BaseLastLayerModel):
    def __init__(self):
        super().__init__()
        self.vit = make_backbone(self.DROP_PATH)
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, prefix_tokens, _, _, _ = self.final_outputs(x)
        feat = prefix_tokens[:, 1:1 + self.NUM_REG, :].mean(dim=1)
        return self.classifier(feat.float()), feat


class PatchViT(BaseLastLayerModel):
    def __init__(self):
        super().__init__()
        self.vit = make_backbone(self.DROP_PATH)
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        spatial_map, _, _, _, _ = self.final_outputs(x)
        feat = spatial_map.flatten(2).mean(dim=2)
        return self.classifier(feat.float()), feat


class PatchAttnViT(BaseLastLayerModel):
    NUM_PATCHES = 256
    NUM_POOL_HEADS = 4

    def __init__(self):
        super().__init__()
        self.vit = make_backbone(self.DROP_PATH)
        self.patch_pool = AttentionPool(
            self.EMBED_DIM,
            num_heads=self.NUM_POOL_HEADS,
            num_patches=self.NUM_PATCHES,
        )
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, _, patch_tokens, _, _ = self.final_outputs(x)
        feat = self.patch_pool(patch_tokens)
        return self.classifier(feat.float()), feat


class LastLayerAllViT(BaseLastLayerModel):
    NUM_PATCHES = 256
    NUM_POOL_HEADS = 4

    def __init__(self):
        super().__init__()
        self.vit = make_backbone(self.DROP_PATH)
        self.patch_pool = AttentionPool(
            self.EMBED_DIM,
            num_heads=self.NUM_POOL_HEADS,
            num_patches=self.NUM_PATCHES,
        )
        self.classifier = nn.Linear(3 * self.EMBED_DIM, 2)

    def forward(self, x):
        _, prefix_tokens, patch_tokens, _, _ = self.final_outputs(x)
        cls_feat = prefix_tokens[:, 0, :]
        reg_feat = prefix_tokens[:, 1:1 + self.NUM_REG, :].mean(dim=1)
        patch_feat = self.patch_pool(patch_tokens)
        fused = torch.cat([cls_feat.float(), reg_feat.float(), patch_feat.float()], dim=1)
        return self.classifier(fused), fused


def clean_state_dict(state):
    state = state.get("state_dict", state.get("model", state))
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def load_checkpoint(model: nn.Module, checkpoint: str, device: torch.device) -> nn.Module:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = clean_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"  loaded {checkpoint_path}")
    return model


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_heatmap(base_image: Image.Image, grid: np.ndarray, out_path: Path, title: str, alpha: float = 0.45):
    grid = normalize_map(grid)
    heat = jet_colormap(grid)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(base_image.size, Image.BICUBIC)
    overlay = Image.blend(base_image.convert("RGB"), heat_img.convert("RGB"), alpha)

    canvas = Image.new("RGB", (overlay.width, overlay.height + 28), "white")
    canvas.paste(overlay, (0, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def patch_tokens_from_model(model: BaseLastLayerModel, image_tensor: torch.Tensor):
    spatial_map, prefix_tokens, patch_tokens, height, width = model.final_outputs(image_tensor)
    return prefix_tokens, patch_tokens, height, width


def cosine_grid(query: torch.Tensor, patches: torch.Tensor, height: int, width: int) -> np.ndarray:
    query = torch.nn.functional.normalize(query.float(), dim=-1)
    patches = torch.nn.functional.normalize(patches.float(), dim=-1)
    scores = (patches * query.unsqueeze(1)).sum(dim=-1)
    return scores[0].detach().cpu().numpy().reshape(height, width)


def save_pool_attention_maps(
    base_image: Image.Image,
    attn: torch.Tensor,
    height: int,
    width: int,
    out_dir: Path,
    prefix: str,
):
    attn_np = attn[0].detach().float().cpu().numpy()  # heads, N
    for head_idx in range(attn_np.shape[0]):
        grid = attn_np[head_idx].reshape(height, width)
        save_heatmap(
            base_image,
            grid,
            out_dir / f"{prefix}_head{head_idx + 1}.png",
            f"{prefix} | attention head {head_idx + 1}",
        )
    mean_grid = attn_np.mean(axis=0).reshape(height, width)
    save_heatmap(base_image, mean_grid, out_dir / f"{prefix}_mean.png", f"{prefix} | attention mean")


@torch.inference_mode()
def run_last_layer_models(args, image_tensor: torch.Tensor, base_image: Image.Image, device: torch.device):
    configs: Dict[str, Tuple[nn.Module, str]] = {
        "cls": (CLSViT(), args.cls_ckpt),
        "reg": (RegViT(), args.reg_ckpt),
        "patch_avg": (PatchViT(), args.patch_ckpt),
        "patch_attn": (PatchAttnViT(), args.patch_attn_ckpt),
        "last_layer_all": (LastLayerAllViT(), args.last_layer_all_ckpt),
    }

    for name, (model, ckpt) in configs.items():
        print(f"\n{name}")
        model = load_checkpoint(model, ckpt, device)
        prefix_tokens, patch_tokens, height, width = patch_tokens_from_model(model, image_tensor)

        out_dir = Path(args.out_dir) / name
        if name == "cls":
            cls_feat = prefix_tokens[:, 0, :]
            grid = cosine_grid(cls_feat, patch_tokens, height, width)
            save_heatmap(base_image, grid, out_dir / "cls_to_patch_similarity.png", "CLS to patch similarity")
        elif name == "reg":
            reg_feat = prefix_tokens[:, 1:1 + model.NUM_REG, :].mean(dim=1)
            grid = cosine_grid(reg_feat, patch_tokens, height, width)
            save_heatmap(base_image, grid, out_dir / "reg_to_patch_similarity.png", "Mean REG to patch similarity")
        elif name == "patch_avg":
            grid = np.ones((height, width), dtype=np.float32)
            save_heatmap(base_image, grid, out_dir / "uniform_avg_pooling.png", "Patch average pooling weights")
        else:
            attn = model.patch_pool.attention(patch_tokens)
            save_pool_attention_maps(base_image, attn, height, width, out_dir, name)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def run_four_layer_model(args, image_tensor: torch.Tensor, base_image: Image.Image, device: torch.device):
    print("\nvit_4layers")
    model = load_checkpoint(FourLayerViT(), args.vit4_ckpt, device)
    _, intermediates = model.vit.forward_intermediates(
        image_tensor,
        indices=model.LAYERS,
        return_prefix_tokens=True,
        norm=True,
    )

    out_dir = Path(args.out_dir) / "vit_4layers"
    for layer_idx, (spatial_map, _) in enumerate(intermediates):
        bsz, channels, height, width = spatial_map.shape
        patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
            bsz, height * width, channels
        )
        attn = model.spatial_heads[layer_idx].patch_pool.attention(patch_tokens)
        layer_name = f"layer{model.LAYERS[layer_idx]}"
        save_pool_attention_maps(base_image, attn, height, width, out_dir, layer_name)


def main():
    parser = argparse.ArgumentParser("Save attention maps for ablation checkpoints")
    parser.add_argument("--image", default="", type=str, help="Optional single input image path.")
    parser.add_argument("--manifest", default="E:/Work/sampled_30k/manifest_onct.csv", type=str)
    parser.add_argument("--root_dir", default="E:/Work/sampled_30k/", type=str)
    parser.add_argument("--num_images", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_dir", default="attention_maps", type=str)
    parser.add_argument("--cls_ckpt", default="checkpoints_cls/best.pth", type=str)
    parser.add_argument("--reg_ckpt", default="checkpoints_reg/best.pth", type=str)
    parser.add_argument("--patch_ckpt", default="checkpoints_patch/best.pth", type=str)
    parser.add_argument("--patch_attn_ckpt", default="checkpoints_patch_attn/best.pth", type=str)
    parser.add_argument("--last_layer_all_ckpt", default="checkpoints_last_layer_all/best.pth", type=str)
    parser.add_argument("--vit4_ckpt", default="checkpoints_vit_4layers/best.pth", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sample_images_from_manifest(
            args.manifest,
            args.root_dir,
            num_images=args.num_images,
            seed=args.seed,
        )
        print(f"Sampled {len(image_paths)} images from {args.manifest}")

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    original_out_dir = args.out_dir
    for idx, image_path in enumerate(image_paths):
        sample_dir = base_out_dir / safe_sample_name(image_path, idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        args.out_dir = str(sample_dir)

        print("\n" + "=" * 88)
        print(f"Image {idx + 1}/{len(image_paths)}: {image_path}")
        print(f"Output: {sample_dir}")
        print("=" * 88)

        image_tensor, base_image = load_image(image_path)
        image_tensor = image_tensor.to(device)
        base_image.save(sample_dir / "input.png")

        run_last_layer_models(args, image_tensor, base_image, device)
        run_four_layer_model(args, image_tensor, base_image, device)

    args.out_dir = original_out_dir
    print(f"\nSaved maps to: {base_out_dir.resolve()}")


if __name__ == "__main__":
    main()
