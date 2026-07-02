"""
Standalone CDFv3 evaluator for last-layer ablation checkpoints.

Evaluates these checkpoints by default:
  - checkpoints_cls/best.pth
  - checkpoints_reg/best.pth
  - checkpoints_patch/best.pth
  - checkpoints_patch_attn/best.pth

The four models match cls.py, reg.py, patch.py, and patch_attn.py, but this
file does not import those training scripts.
"""

import argparse
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_and_normalize(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def video_id_from_sample_dir(sample_dir: str) -> str:
    return Path(sample_dir).parent.name


class CDFv3FrameDataset(Dataset):
    """
    CDFv3 manifest convention is assumed to be label=1 for real and label=0
    for fake. Models are trained with label=0 real and label=1 fake, so labels
    are remapped here.
    """

    def __init__(self, csv_path: str, root_dir: str, real_label: int = 1):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        required = {"sample_dir", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns {missing}; found {list(df.columns)}")

        df["label"] = df["label"].astype(int)
        root = Path(root_dir)

        paths = df["sample_dir"].apply(lambda rel: str(root / rel / "image.png")).values
        labels_raw = df["label"].values
        video_ids = df["sample_dir"].apply(video_id_from_sample_dir).values

        labels = np.array([0 if int(y) == real_label else 1 for y in labels_raw], dtype=np.int64)
        exists = np.array([os.path.isfile(p) for p in paths])
        skipped = int((~exists).sum())
        if skipped:
            print(f"  [CDFv3] skipped {skipped} missing image.png files")

        self.entries = list(zip(paths[exists], labels[exists], video_ids[exists]))
        kept_labels = labels[exists]
        print(
            f"  CDFv3 frames: Real={int((kept_labels == 0).sum())} | "
            f"Fake={int((kept_labels == 1).sum())} | Total={len(self.entries)}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, video_id = self.entries[idx]
        return load_and_normalize(path), int(label), video_id


class BaseDINOv3Ablation(nn.Module):
    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_REG = 4

    def make_backbone(self):
        vit = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
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

    def final_tokens(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        return intermediates[0]


class CLSViT(BaseDINOv3Ablation):
    def __init__(self):
        super().__init__()
        self.vit = self.make_backbone()
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, prefix_tokens = self.final_tokens(x)
        feat = prefix_tokens[:, 0, :]
        logits = self.classifier(feat.float())
        return logits, feat


class RegViT(BaseDINOv3Ablation):
    def __init__(self):
        super().__init__()
        self.vit = self.make_backbone()
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        _, prefix_tokens = self.final_tokens(x)
        feat = prefix_tokens[:, 1:1 + self.NUM_REG, :].mean(dim=1)
        logits = self.classifier(feat.float())
        return logits, feat


class PatchViT(BaseDINOv3Ablation):
    def __init__(self):
        super().__init__()
        self.vit = self.make_backbone()
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        spatial_map, _ = self.final_tokens(x)
        feat = spatial_map.flatten(2).mean(dim=2)
        logits = self.classifier(feat.float())
        return logits, feat


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        x_heads = x.view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.query.expand(bsz, -1, -1, -1)
        attn = (q @ x_heads.transpose(-2, -1)) * self.scale
        attn = attn + self._positional_bias(num_tokens, x.device, attn.dtype)
        attn = attn.softmax(dim=-1)
        out = attn @ x_heads
        return out.permute(0, 2, 1, 3).reshape(bsz, channels)


class PatchAttnViT(BaseDINOv3Ablation):
    NUM_PATCHES = 256
    NUM_POOL_HEADS = 4

    def __init__(self):
        super().__init__()
        self.vit = self.make_backbone()
        self.patch_pool = AttentionPool(
            self.EMBED_DIM,
            num_heads=self.NUM_POOL_HEADS,
            num_patches=self.NUM_PATCHES,
        )
        self.classifier = nn.Linear(self.EMBED_DIM, 2)

    def forward(self, x):
        spatial_map, _ = self.final_tokens(x)
        bsz, channels, height, width = spatial_map.shape
        patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
            bsz, height * width, channels
        )
        feat = self.patch_pool(patch_tokens)
        logits = self.classifier(feat.float())
        return logits, feat


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model", obj)) if isinstance(obj, dict) else obj
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def load_model(model_cls, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = model_cls().to(device)
    state = clean_state_dict(torch.load(ckpt_path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"  missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"  loaded checkpoint: {ckpt_path}")
    return model


def compute_binary_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def video_mean_metrics(labels, probs, video_ids):
    grouped_probs = defaultdict(list)
    grouped_labels = {}
    for label, prob, video_id in zip(labels, probs, video_ids):
        grouped_probs[video_id].append(float(prob))
        grouped_labels[video_id] = int(label)

    video_labels = []
    video_probs = []
    for video_id in sorted(grouped_probs):
        video_labels.append(grouped_labels[video_id])
        video_probs.append(float(np.mean(grouped_probs[video_id])))
    return compute_binary_metrics(video_labels, video_probs), len(video_labels)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, name: str):
    model.eval()
    labels_all, probs_all, video_ids_all = [], [], []
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with amp_ctx:
        for imgs, labels, video_ids in tqdm(loader, desc=f"CDFv3 | {name}", leave=True):
            imgs = imgs.to(device, non_blocking=True)
            logits, _ = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().tolist())
            video_ids_all.extend(list(video_ids))

    frame_metrics = compute_binary_metrics(labels_all, probs_all)
    vid_metrics, num_videos = video_mean_metrics(labels_all, probs_all, video_ids_all)
    return frame_metrics, vid_metrics, len(labels_all), num_videos


def print_metric_line(prefix: str, metrics: dict, count_name: str, count: int):
    print(
        f"{prefix:18s} | {count_name}={count:6d} | "
        f"AUC={metrics['auc'] * 100:6.2f} | "
        f"AP={metrics['ap'] * 100:6.2f} | "
        f"Acc={metrics['acc'] * 100:6.2f} | "
        f"F1={metrics['f1'] * 100:6.2f}"
    )


def main():
    parser = argparse.ArgumentParser("Standalone CDFv3 test for last-layer ablation checkpoints")
    parser.add_argument("--cdfv3_root", required=True, type=str)
    parser.add_argument("--cdfv3_csv", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--real_label", default=1, type=int, help="CDFv3 label value for real videos/frames.")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--cls_ckpt", default="checkpoints_cls/best.pth", type=str)
    parser.add_argument("--reg_ckpt", default="checkpoints_reg/best.pth", type=str)
    parser.add_argument("--patch_ckpt", default="checkpoints_patch/best.pth", type=str)
    parser.add_argument("--patch_attn_ckpt", default="checkpoints_patch_attn/best.pth", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CDFv3 root: {args.cdfv3_root}")
    print(f"CDFv3 csv : {args.cdfv3_csv}")
    print(f"Label mapping: manifest real_label={args.real_label} -> model label 0; others -> model label 1")

    dataset = CDFv3FrameDataset(args.cdfv3_csv, args.cdfv3_root, real_label=args.real_label)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    experiments = [
        ("CLS", CLSViT, args.cls_ckpt),
        ("REG", RegViT, args.reg_ckpt),
        ("Patch", PatchViT, args.patch_ckpt),
        ("Patch*", PatchAttnViT, args.patch_attn_ckpt),
    ]

    summary = []
    for name, model_cls, ckpt_path in experiments:
        print("\n" + "=" * 88)
        print(f"Testing {name} checkpoint on CDFv3")
        print("=" * 88)
        model = load_model(model_cls, ckpt_path, device)

        if not args.no_compile and hasattr(torch, "compile"):
            print("  compiling model with torch.compile ...")
            model = torch.compile(model)

        frame_metrics, vid_metrics, num_frames, num_videos = evaluate(model, loader, device, name)
        print_metric_line(f"{name} frame", frame_metrics, "frames", num_frames)
        print_metric_line(f"{name} video-mean", vid_metrics, "videos", num_videos)
        summary.append((name, frame_metrics, vid_metrics))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("CDFv3 Summary")
    print("=" * 88)
    for name, frame_metrics, vid_metrics in summary:
        print(
            f"{name:8s} | "
            f"Frame AUC={frame_metrics['auc'] * 100:6.2f} | "
            f"Video-mean AUC={vid_metrics['auc'] * 100:6.2f}"
        )


if __name__ == "__main__":
    main()
