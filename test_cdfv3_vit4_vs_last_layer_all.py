"""
Evaluate checkpoints_vit_4layers and checkpoints_last_layer_all on CDFv3.

For checkpoints_vit_4layers, the deepest tapped layer logits are used
(index 3, corresponding to DINOv3 block 23).

For checkpoints_last_layer_all, the single final-layer classifier output is
used directly.
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

from frame_model import ViT as FourLayerViT


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
    CDFv3 manifest is assumed to use label=1 for real and label=0 for fake.
    The model convention is label=0 real and label=1 fake, so labels are
    remapped here.
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
        kept = labels[exists]
        print(
            f"  CDFv3 frames: Real={int((kept == 0).sum())} | "
            f"Fake={int((kept == 1).sum())} | Total={len(self.entries)}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, video_id = self.entries[idx]
        return load_and_normalize(path), int(label), video_id


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        x_heads = x.view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.query.expand(bsz, -1, -1, -1)
        attn = (q @ x_heads.transpose(-2, -1)) * self.scale
        attn = attn + self._positional_bias(num_tokens, x.device, attn.dtype)
        attn = attn.softmax(dim=-1)
        out = attn @ x_heads
        return out.permute(0, 2, 1, 3).reshape(bsz, channels)


class LastLayerAllViT(nn.Module):
    EMBED_DIM = 1024
    NUM_REG = 4
    FINAL_LAYER = 23
    DROP_PATH = 0.10
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
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        spatial_map, prefix_tokens = intermediates[0]
        cls_feat = prefix_tokens[:, 0, :]
        reg_feat = prefix_tokens[:, 1:1 + self.NUM_REG, :].mean(dim=1)

        bsz, channels, height, width = spatial_map.shape
        patch_tokens = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(
            bsz, height * width, channels
        )
        patch_feat = self.patch_pool(patch_tokens)
        fused = torch.cat([cls_feat.float(), reg_feat.float(), patch_feat.float()], dim=1)
        logits = self.classifier(fused)
        return logits


def clean_state_dict(state):
    state = state.get("state_dict", state.get("model", state))
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def checkpoint_path(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        p = p / "best.pth"
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return str(p)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    ckpt = checkpoint_path(path)
    state = clean_state_dict(torch.load(ckpt, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"  loaded checkpoint: {ckpt}")
    return model


def compute_metrics(labels, probs):
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

    video_labels, video_probs = [], []
    for video_id in sorted(grouped_probs):
        video_labels.append(grouped_labels[video_id])
        video_probs.append(float(np.mean(grouped_probs[video_id])))
    return compute_metrics(video_labels, video_probs), len(video_labels)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, name: str, kind: str):
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
            if kind == "vit4":
                logits_list, _, _ = model(imgs)
                logits = logits_list[3]
            elif kind == "last_layer_all":
                logits = model(imgs)
            else:
                raise ValueError(f"Unknown model kind: {kind}")

            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().tolist())
            video_ids_all.extend(list(video_ids))

    frame_metrics = compute_metrics(labels_all, probs_all)
    vid_metrics, num_videos = video_mean_metrics(labels_all, probs_all, video_ids_all)
    return frame_metrics, vid_metrics, len(labels_all), num_videos


def print_metrics(name: str, frame_metrics: dict, vid_metrics: dict, num_frames: int, num_videos: int):
    print(
        f"{name:18s} frame      | frames={num_frames:6d} | "
        f"AUC={frame_metrics['auc'] * 100:6.2f} | AP={frame_metrics['ap'] * 100:6.2f} | "
        f"Acc={frame_metrics['acc'] * 100:6.2f} | F1={frame_metrics['f1'] * 100:6.2f}"
    )
    print(
        f"{name:18s} video-mean | videos={num_videos:6d} | "
        f"AUC={vid_metrics['auc'] * 100:6.2f} | AP={vid_metrics['ap'] * 100:6.2f} | "
        f"Acc={vid_metrics['acc'] * 100:6.2f} | F1={vid_metrics['f1'] * 100:6.2f}"
    )


def main():
    parser = argparse.ArgumentParser("Test checkpoints_vit_4layers and checkpoints_last_layer_all on CDFv3")
    parser.add_argument("--cdfv3_root", required=True, type=str)
    parser.add_argument("--cdfv3_csv", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--real_label", default=1, type=int)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--vit4_ckpt", default="checkpoints_vit_4layers/best.pth", type=str)
    parser.add_argument("--last_layer_all_ckpt", default="checkpoints_last_layer_all/best.pth", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CDFv3 root: {args.cdfv3_root}")
    print(f"CDFv3 csv : {args.cdfv3_csv}")
    print(f"Label mapping: manifest real_label={args.real_label} -> model real=0; others -> fake=1")

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
        ("vit_4layers_l23", FourLayerViT(), args.vit4_ckpt, "vit4"),
        ("last_layer_all", LastLayerAllViT(), args.last_layer_all_ckpt, "last_layer_all"),
    ]

    summary = []
    for name, model, ckpt, kind in experiments:
        print("\n" + "=" * 88)
        print(f"Testing {name}")
        print("=" * 88)
        model = load_checkpoint(model, ckpt, device)
        if not args.no_compile and hasattr(torch, "compile"):
            print("  compiling model with torch.compile ...")
            model = torch.compile(model)

        frame_metrics, vid_metrics, num_frames, num_videos = evaluate(model, loader, device, name, kind)
        print_metrics(name, frame_metrics, vid_metrics, num_frames, num_videos)
        summary.append((name, frame_metrics, vid_metrics))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("CDFv3 Summary")
    print("=" * 88)
    for name, frame_metrics, vid_metrics in summary:
        print(
            f"{name:18s} | Frame AUC={frame_metrics['auc'] * 100:6.2f} | "
            f"Video-mean AUC={vid_metrics['auc'] * 100:6.2f}"
        )


if __name__ == "__main__":
    main()
