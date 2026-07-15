"""
Mixed-curvature CLS-token training in a product manifold.

This follows the product-manifold idea from "Learning Mixed-Curvature
Representations in Products of Model Spaces" more closely than the separate
spherical/hyperbolic classifier scripts:

  CLS -> projector -> H^d x ... x S^d x ... x E^d

Training uses the product geodesic distance:

  d_P(x, y)^2 = sum_i d_i(x_i, y_i)^2

and a distortion loss that matches batch pair distances to a simple supervised
target metric. A small prototype-distance CE term is kept by default so the
embedding remains directly usable as a real/fake detector. Set
--prototype_weight 0 to train only with the product distortion loss.

Protocol is kept aligned with hyperspherical_cls.py / hyperbolic_cls.py:
  - train on FF++ frames from manifest/root
  - tiny FF++ validation split
  - test on CDFv1 every epoch
  - same DINOv3 final CLS token, LoRA setup, augmentations, metrics, and saving
"""

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.optim as optim
import transformers

if not hasattr(transformers, "HybridCache"):
    transformers.HybridCache = getattr(transformers, "DynamicCache", getattr(transformers, "Cache", object))

from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentations import augment_batch, load_and_resize, normalize


parser = argparse.ArgumentParser(
    description="DINOv3 final-CLS training in a mixed-curvature product manifold."
)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=36, type=int)
parser.add_argument("--save_root", default="checkpoints_mixed_curvature_cls", type=str)
parser.add_argument("--load_from", default="checkpoints_spherical_cls/best.pth", type=str)
parser.add_argument("--no_compile", action="store_true")

parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup_steps", default=512, type=int)
parser.add_argument("--lora_r", default=32, type=int)
parser.add_argument("--lora_alpha", default=64, type=int)
parser.add_argument("--lora_dropout", default=0.10, type=float)
parser.add_argument("--max_train_samples", default=0, type=int)
parser.add_argument("--max_val_samples", default=0, type=int)
parser.add_argument("--max_frames_per_dataset", default=0, type=int)
parser.add_argument("--val_ratio", default=0.005, type=float)
parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--hyperbolic_dims", default="64,64", type=str,
                    help="Comma-separated tangent dimensions for hyperboloid factors. Empty disables them.")
parser.add_argument("--sphere_dims", default="64", type=str,
                    help="Comma-separated tangent dimensions for spherical factors. Empty disables them.")
parser.add_argument("--euclidean_dim", default=128, type=int)
parser.add_argument("--init_radius", default=1.0, type=float,
                    help="Initial radius scale for non-Euclidean product factors.")
parser.add_argument("--min_radius", default=0.05, type=float)
parser.add_argument("--fixed_curvature", action="store_true",
                    help="Freeze manifold radii instead of learning curvature scales.")
parser.add_argument("--prototype_tangent_norm", default=1.0, type=float,
                    help="Class prototype distance from origin/north pole in tangent coordinates.")
parser.add_argument("--euclidean_proto_radius", default=1.0, type=float)
parser.add_argument("--max_hyperbolic_norm", default=4.0, type=float,
                    help="Stable cap for hyperboloid expmap tangent norm.")
parser.add_argument("--max_sphere_angle", default=2.6, type=float,
                    help="Stable cap for sphere expmap tangent norm, in radians.")
parser.add_argument("--same_label_distance", default=0.5, type=float)
parser.add_argument("--different_label_distance", default=2.0, type=float)
parser.add_argument("--prototype_weight", default=0.10, type=float,
                    help="Auxiliary CE weight on product-distance class prototypes.")

parser.add_argument("--manifest",
                    default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv",
                    type=str)
parser.add_argument("--root_dir",
                    default="/media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out",
                    type=str)
parser.add_argument("--cdf_root", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv", default="/media/tarun/B482367C823642E2/usr/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
args = parser.parse_args()


IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers
torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


def parse_dims(text: str) -> list[int]:
    if text is None or not str(text).strip():
        return []
    dims = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if any(dim < 1 for dim in dims):
        raise ValueError(f"All product factor dimensions must be >= 1. Got: {text}")
    return dims


def inverse_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


def simplex_directions(num_classes: int, dim: int, name: str) -> torch.Tensor:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if dim < num_classes:
        raise ValueError(f"{name} must be >= num_classes")
    eye = torch.eye(num_classes, dtype=torch.float32)
    simplex = eye - eye.mean(dim=0, keepdim=True)
    simplex = nn.functional.normalize(simplex, dim=1)
    if dim > num_classes:
        simplex = torch.cat([simplex, torch.zeros(num_classes, dim - num_classes)], dim=1)
    return simplex


def stable_scaled_norm(norm: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return norm
    return float(max_norm) * torch.tanh(norm / float(max_norm))


def hyperboloid_expmap0(v: torch.Tensor, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Unit-curvature hyperboloid expmap at origin.

    Tangent dim d -> ambient dim d+1 with Minkowski metric
    <-,+,...,+>. The point satisfies -x0^2 + ||x_spatial||^2 = -1.
    """
    v = v.float()
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = stable_scaled_norm(norm, max_norm)
    direction = v / norm
    time = torch.cosh(theta)
    spatial = torch.sinh(theta) * direction
    return torch.cat([time, spatial], dim=-1)


def sphere_expmap_north(v: torch.Tensor, max_angle: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Unit sphere expmap at the north pole.

    Tangent dim d -> ambient dim d+1. The output has unit Euclidean norm.
    """
    v = v.float()
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = stable_scaled_norm(norm, max_angle)
    direction = v / norm
    north = torch.cos(theta)
    spatial = torch.sin(theta) * direction
    return torch.cat([north, spatial], dim=-1)


def acosh_safe(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.acosh(x.clamp_min(1.0 + eps))


class MixedCurvatureHead(nn.Module):
    """Project CLS features into a product of hyperbolic, spherical, and Euclidean factors."""

    def __init__(
        self,
        in_dim: int = 1024,
        hyperbolic_dims: list[int] | None = None,
        sphere_dims: list[int] | None = None,
        euclidean_dim: int = 128,
        num_classes: int = 2,
        init_radius: float = 1.0,
        min_radius: float = 0.05,
        learn_curvature: bool = True,
        prototype_tangent_norm: float = 1.0,
        euclidean_proto_radius: float = 1.0,
        max_hyperbolic_norm: float = 4.0,
        max_sphere_angle: float = 2.6,
    ):
        super().__init__()
        self.hyperbolic_dims = list(hyperbolic_dims or [])
        self.sphere_dims = list(sphere_dims or [])
        self.euclidean_dim = int(euclidean_dim)
        self.num_classes = int(num_classes)
        self.min_radius = float(min_radius)
        self.prototype_tangent_norm = float(prototype_tangent_norm)
        self.euclidean_proto_radius = float(euclidean_proto_radius)
        self.max_hyperbolic_norm = float(max_hyperbolic_norm)
        self.max_sphere_angle = float(max_sphere_angle)

        if self.euclidean_dim < 0:
            raise ValueError("--euclidean_dim must be >= 0")
        if not self.hyperbolic_dims and not self.sphere_dims and self.euclidean_dim == 0:
            raise ValueError("At least one product factor is required.")

        tangent_dim = sum(self.hyperbolic_dims) + sum(self.sphere_dims) + self.euclidean_dim
        self.tangent_dim = tangent_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, tangent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(tangent_dim, tangent_dim),
        )

        radius_init = inverse_softplus(max(float(init_radius) - self.min_radius, 1e-4))
        self.hyper_radius_raw = nn.Parameter(torch.full((len(self.hyperbolic_dims),), radius_init))
        self.sphere_radius_raw = nn.Parameter(torch.full((len(self.sphere_dims),), radius_init))
        if not learn_curvature:
            self.hyper_radius_raw.requires_grad_(False)
            self.sphere_radius_raw.requires_grad_(False)

        for idx, dim in enumerate(self.hyperbolic_dims):
            self.register_buffer(f"hyper_dirs_{idx}", simplex_directions(num_classes, dim, f"hyperbolic_dims[{idx}]"))
        for idx, dim in enumerate(self.sphere_dims):
            self.register_buffer(f"sphere_dirs_{idx}", simplex_directions(num_classes, dim, f"sphere_dims[{idx}]"))
        if self.euclidean_dim > 0:
            self.register_buffer(
                "euclidean_dirs",
                simplex_directions(num_classes, self.euclidean_dim, "euclidean_dim"),
            )

    def hyper_radii(self) -> torch.Tensor:
        return nn.functional.softplus(self.hyper_radius_raw.float()) + self.min_radius

    def sphere_radii(self) -> torch.Tensor:
        return nn.functional.softplus(self.sphere_radius_raw.float()) + self.min_radius

    def split_tangent(self, tangent: torch.Tensor):
        cursor = 0
        hyper_tangent = []
        sphere_tangent = []
        for dim in self.hyperbolic_dims:
            hyper_tangent.append(tangent[:, cursor:cursor + dim])
            cursor += dim
        for dim in self.sphere_dims:
            sphere_tangent.append(tangent[:, cursor:cursor + dim])
            cursor += dim
        euclidean = None
        if self.euclidean_dim > 0:
            euclidean = tangent[:, cursor:cursor + self.euclidean_dim].float()
            cursor += self.euclidean_dim
        return hyper_tangent, sphere_tangent, euclidean

    def project_components(self, tangent: torch.Tensor):
        hyper_tangent, sphere_tangent, euclidean = self.split_tangent(tangent)
        hyper = [
            hyperboloid_expmap0(block, self.max_hyperbolic_norm)
            for block in hyper_tangent
        ]
        sphere = [
            sphere_expmap_north(block, self.max_sphere_angle)
            for block in sphere_tangent
        ]
        return {"hyper": hyper, "sphere": sphere, "euclidean": euclidean}

    def class_prototypes(self, device):
        hyper = []
        sphere = []
        for idx in range(len(self.hyperbolic_dims)):
            dirs = getattr(self, f"hyper_dirs_{idx}").to(device)
            tangent = dirs * self.prototype_tangent_norm
            hyper.append(hyperboloid_expmap0(tangent, self.max_hyperbolic_norm))
        for idx in range(len(self.sphere_dims)):
            dirs = getattr(self, f"sphere_dirs_{idx}").to(device)
            tangent = dirs * self.prototype_tangent_norm
            sphere.append(sphere_expmap_north(tangent, self.max_sphere_angle))
        euclidean = None
        if self.euclidean_dim > 0:
            euclidean = self.euclidean_dirs.to(device).float() * self.euclidean_proto_radius
        return {"hyper": hyper, "sphere": sphere, "euclidean": euclidean}

    def product_pairwise_distance_sq(self, components) -> torch.Tensor:
        batch_size = None
        for key in ("hyper", "sphere"):
            if components[key]:
                batch_size = components[key][0].shape[0]
                break
        if batch_size is None:
            batch_size = components["euclidean"].shape[0]
        out = torch.zeros(batch_size, batch_size, device=self.hyper_radius_raw.device, dtype=torch.float32)

        radii_h = self.hyper_radii()
        for idx, x in enumerate(components["hyper"]):
            x = x.float()
            neg_minkowski_inner = x[:, :1] @ x[:, :1].T - x[:, 1:] @ x[:, 1:].T
            dist = radii_h[idx] * acosh_safe(neg_minkowski_inner)
            out = out + dist.pow(2)

        radii_s = self.sphere_radii()
        for idx, x in enumerate(components["sphere"]):
            x = x.float()
            cos = (x @ x.T).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
            dist = radii_s[idx] * torch.acos(cos)
            out = out + dist.pow(2)

        if components["euclidean"] is not None:
            out = out + torch.cdist(components["euclidean"].float(), components["euclidean"].float()).pow(2)
        return out

    def product_distance_to_prototypes_sq(self, components) -> torch.Tensor:
        prototypes = self.class_prototypes(self.hyper_radius_raw.device)
        batch_size = None
        for key in ("hyper", "sphere"):
            if components[key]:
                batch_size = components[key][0].shape[0]
                break
        if batch_size is None:
            batch_size = components["euclidean"].shape[0]
        out = torch.zeros(batch_size, self.num_classes, device=self.hyper_radius_raw.device, dtype=torch.float32)

        radii_h = self.hyper_radii()
        for idx, x in enumerate(components["hyper"]):
            x = x.float()
            p = prototypes["hyper"][idx].float()
            neg_minkowski_inner = x[:, :1] @ p[:, :1].T - x[:, 1:] @ p[:, 1:].T
            dist = radii_h[idx] * acosh_safe(neg_minkowski_inner)
            out = out + dist.pow(2)

        radii_s = self.sphere_radii()
        for idx, x in enumerate(components["sphere"]):
            x = x.float()
            p = prototypes["sphere"][idx].float()
            cos = (x @ p.T).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
            dist = radii_s[idx] * torch.acos(cos)
            out = out + dist.pow(2)

        if components["euclidean"] is not None:
            out = out + torch.cdist(components["euclidean"].float(), prototypes["euclidean"].float()).pow(2)
        return out

    def forward(self, x: torch.Tensor):
        tangent = self.projector(x.float())
        components = self.project_components(tangent)
        proto_dist_sq = self.product_distance_to_prototypes_sq(components)
        logits = -proto_dist_sq
        return logits, components, proto_dist_sq

    def radius_summary(self) -> str:
        with torch.no_grad():
            h = self.hyper_radii().detach().cpu().numpy().round(4).tolist()
            s = self.sphere_radii().detach().cpu().numpy().round(4).tolist()
        return f"hyper_radii={h} sphere_radii={s}"


class MixedCurvatureCLSViT(nn.Module):
    EMBED_DIM = 1024
    FINAL_LAYER = 23
    DROP_PATH = 0.10
    NUM_CLASSES = 2

    def __init__(
        self,
        hyperbolic_dims: list[int],
        sphere_dims: list[int],
        euclidean_dim: int,
        init_radius: float,
        min_radius: float,
        learn_curvature: bool,
        prototype_tangent_norm: float,
        euclidean_proto_radius: float,
        max_hyperbolic_norm: float,
        max_sphere_angle: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["attn.qkv"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)
        self.mixed_head = MixedCurvatureHead(
            in_dim=self.EMBED_DIM,
            hyperbolic_dims=hyperbolic_dims,
            sphere_dims=sphere_dims,
            euclidean_dim=euclidean_dim,
            num_classes=self.NUM_CLASSES,
            init_radius=init_radius,
            min_radius=min_radius,
            learn_curvature=learn_curvature,
            prototype_tangent_norm=prototype_tangent_norm,
            euclidean_proto_radius=euclidean_proto_radius,
            max_hyperbolic_norm=max_hyperbolic_norm,
            max_sphere_angle=max_sphere_angle,
        )

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=[self.FINAL_LAYER],
            return_prefix_tokens=True,
            norm=True,
        )
        _, prefix_tokens = intermediates[0]
        cls = prefix_tokens[:, 0, :]
        return self.mixed_head(cls)


def mixed_curvature_loss(
    model,
    logits,
    components,
    labels,
    same_label_distance: float,
    different_label_distance: float,
    prototype_weight: float,
):
    labels = labels.long()
    pair_dist_sq = model.mixed_head.product_pairwise_distance_sq(components)
    batch_size = labels.numel()
    if batch_size <= 1:
        raise ValueError("mixed_curvature_loss needs batch_size > 1")

    same = labels[:, None].eq(labels[None, :])
    targets = torch.where(
        same,
        torch.full_like(pair_dist_sq, float(same_label_distance)),
        torch.full_like(pair_dist_sq, float(different_label_distance)),
    )
    mask = torch.triu(torch.ones(batch_size, batch_size, device=labels.device, dtype=torch.bool), diagonal=1)
    distortion = (pair_dist_sq[mask] / targets[mask].pow(2).clamp_min(1e-6) - 1.0).abs().mean()

    proto_ce = nn.functional.cross_entropy(logits.float(), labels)
    loss = distortion + float(prototype_weight) * proto_ce
    return loss, distortion.detach(), proto_ce.detach()


def print_dataset_item_counts(title: str, items: list[dict]):
    print(f"\n{title}:")
    by_dataset = defaultdict(lambda: [0, 0, 0])
    for item in items:
        counts = by_dataset[item["dataset"]]
        counts[0] += 1
        counts[1 + int(item["label"])] += 1
    total = [0, 0, 0]
    for dataset in sorted(by_dataset):
        counts = by_dataset[dataset]
        total = [a + b for a, b in zip(total, counts)]
        print(f"  {dataset:<8} frames={counts[0]:>8} real={counts[1]:>8} fake={counts[2]:>8}")
    print(f"  {'TOTAL':<8} frames={total[0]:>8} real={total[1]:>8} fake={total[2]:>8}")


def cap_items_by_label(items: list[dict], cap: int, seed: int, split_name: str) -> list[dict]:
    if cap <= 0 or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    capped = []
    for label in [0, 1]:
        class_items = [item for item in items if item["label"] == label]
        n = min(len(class_items), cap // 2 if label == 0 else cap - len(capped))
        if n > 0:
            idx = rng.choice(len(class_items), size=n, replace=False)
            capped.extend(class_items[int(i)] for i in idx)
    if not capped:
        idx = rng.choice(len(items), size=cap, replace=False)
        capped = [items[int(i)] for i in idx]
    order = rng.permutation(len(capped))
    capped = [capped[int(i)] for i in order]
    print_dataset_item_counts(f"{split_name} capped", capped)
    return capped


def split_items_by_label(items: list[dict], val_ratio: float, seed: int):
    if val_ratio <= 0:
        return items, []
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(set(item["label"] for item in items)):
        class_items = [item for item in items if item["label"] == label]
        if len(class_items) <= 1:
            train.extend(class_items)
            continue
        order = rng.permutation(len(class_items))
        class_items = [class_items[int(i)] for i in order]
        val_n = max(1, int(len(class_items) * val_ratio))
        val_n = min(val_n, len(class_items) - 1)
        val.extend(class_items[:val_n])
        train.extend(class_items[val_n:])
    return train, val


def build_ffpp_items(manifest_csv: str, root_dir: str):
    if not manifest_csv or not root_dir:
        raise ValueError("--manifest and --root_dir are required for FF++ training.")
    manifest = Path(manifest_csv)
    root = Path(root_dir)
    if not manifest.is_file():
        raise FileNotFoundError(f"FF++ manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"FF++ root not found: {root}")

    df = pd.read_csv(manifest)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"FF++ manifest must contain {required}. Found: {list(df.columns)}")

    items, skipped = [], 0
    for rel, label in zip(df["sample_dir"].astype(str).str.replace("\\", "/", regex=False), df["label"]):
        rel_path = Path(rel)
        path = rel_path / "image.png" if rel_path.is_absolute() else root / rel / "image.png"
        if path.is_file():
            items.append({"path": str(path), "label": int(label), "dataset": "FFPP"})
        else:
            skipped += 1
    if skipped:
        print(f"  [FFPP] skipped {skipped} missing image.png files")
    print_dataset_item_counts("FF++ frames loaded", items)
    return items


def prepare_splits(args):
    items = build_ffpp_items(args.manifest, args.root_dir)
    if not items:
        raise ValueError("No FF++ training frames found. Check --manifest / --root_dir.")
    train_items, val_items = split_items_by_label(items, args.val_ratio, args.seed + 1000)
    train_items = cap_items_by_label(train_items, args.max_frames_per_dataset, args.seed + 2000, "FFPP train")
    train_items = cap_items_by_label(train_items, args.max_train_samples, args.seed + 9999, "Combined train")
    val_items = cap_items_by_label(val_items, args.max_val_samples, args.seed + 10099, "Combined val")
    if not train_items:
        raise ValueError("No training frames found after split/caps.")
    if not val_items:
        raise ValueError("No validation frames found. Increase --val_ratio or check FF++ paths.")
    rng = np.random.default_rng(args.seed)
    train_items = [train_items[int(i)] for i in rng.permutation(len(train_items))]
    val_items = [val_items[int(i)] for i in rng.permutation(len(val_items))]
    print_dataset_item_counts("Combined train frames", train_items)
    print_dataset_item_counts("Combined val frames", val_items)
    return train_items, val_items


class FrameItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return normalize(load_and_resize(item["path"], IMG_SIZE)), int(item["label"])


class CDFv1Dataset(Dataset):
    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = df["label"].astype(int)
        root = Path(data_root)
        paths = df["sample_dir"].apply(lambda d: str(root / d / "image.png"))
        labels = df["label"].values
        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [CDFv1] skipped {skipped} missing image.png")
        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))
        print(f"CDFv1 eval frames: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        return normalize(load_and_resize(img_path, IMG_SIZE)), int(label)


def compute_metrics(all_labels, all_probs, split_name: str, epoch: int):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f"  [{split_name}] Epoch {epoch+1:02d} | "
          f"AUC={auc:.4f} AP={ap:.4f} Acc={acc*100:.2f}% F1={f1:.4f} EER={eer*100:.2f}% "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return auc


def run_eval(model, loader, desc):
    labels_out, probs_out = [], []
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits, _, _ = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            probs_out.extend(probs.tolist())
            labels_out.extend(labels.numpy().tolist())
    return labels_out, probs_out


def clean_state_dict(state):
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_training_checkpoint(path: str):
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        candidates = [
            ckpt_path / "best.pth",
            ckpt_path / "latest.pth",
            ckpt_path / "best_resume.pth",
            ckpt_path / "latest_resume.pth",
        ]
        ckpt_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        model_state = clean_state_dict(raw["model"])
        train_state = raw
    elif isinstance(raw, dict) and "state_dict" in raw:
        model_state = clean_state_dict(raw["state_dict"])
        train_state = raw
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        model_state = clean_state_dict(raw["model_state_dict"])
        train_state = raw
    else:
        model_state = clean_state_dict(raw)
        train_state = {}
    return ckpt_path, model_state, train_state


def can_resume_training_state(train_state) -> bool:
    return isinstance(train_state, dict) and train_state.get("loss") == "mixed-curvature product distortion" and "epoch" in train_state


def cdfv1_available(cdf_csv: str, cdf_root: str) -> bool:
    if not cdf_csv or not cdf_root:
        print("CDFv1 eval disabled: --cdf_csv / --cdf_root not provided.")
        return False
    if not Path(cdf_csv).is_file():
        print(f"CDFv1 eval disabled: missing manifest {cdf_csv}")
        return False
    if not Path(cdf_root).is_dir():
        print(f"CDFv1 eval disabled: missing root {cdf_root}")
        return False
    return True


def make_resume_checkpoint(model, optimizer, scheduler, scaler, epoch, best_auc, best_epoch):
    live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "epoch": epoch,
        "model": live_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_test_auc": best_auc,
        "best_epoch": best_epoch,
        "args": vars(args),
        "loss": "mixed-curvature product distortion",
    }


if __name__ == "__main__":
    SEP = "=" * 80
    hyperbolic_dims = parse_dims(args.hyperbolic_dims)
    sphere_dims = parse_dims(args.sphere_dims)
    print(
        "Product signature: "
        f"H{hyperbolic_dims or []} x S{sphere_dims or []} x E^{args.euclidean_dim}"
    )

    train_items, val_items = prepare_splits(args)
    train_dataset = FrameItemDataset(train_items)
    val_dataset = FrameItemDataset(val_items)
    cdf_dataset = CDFv1Dataset(args.cdf_csv, args.cdf_root) if cdfv1_available(args.cdf_csv, args.cdf_root) else None

    _persistent = _num_workers > 0
    _prefetch = 4 if _num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=_num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=_num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    cdf_loader = None
    if cdf_dataset is not None:
        cdf_loader = DataLoader(
            cdf_dataset,
            batch_size=args.batch_size,
            num_workers=_num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
        )

    os.makedirs(args.save_root, exist_ok=True)
    model = MixedCurvatureCLSViT(
        hyperbolic_dims=hyperbolic_dims,
        sphere_dims=sphere_dims,
        euclidean_dim=args.euclidean_dim,
        init_radius=args.init_radius,
        min_radius=args.min_radius,
        learn_curvature=not args.fixed_curvature,
        prototype_tangent_norm=args.prototype_tangent_norm,
        euclidean_proto_radius=args.euclidean_proto_radius,
        max_hyperbolic_norm=args.max_hyperbolic_norm,
        max_sphere_angle=args.max_sphere_angle,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    train_state = {}
    start_epoch = 0
    if args.load_from:
        ckpt_path, model_state, train_state = load_training_checkpoint(args.load_from)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Loaded model weights from {ckpt_path}")
        print(f"  missing keys   : {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler()
    lr_base = args.lr
    epochs = args.epochs
    iter_per_epoch = len(train_loader)
    total_steps = max(epochs * iter_per_epoch, 1)
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 1))
    lr_min = 1e-6 / lr_base
    lr_dict = {
        i: (
            (((1 + math.cos((i - warmup_steps) * math.pi / max(total_steps - warmup_steps, 1))) / 2) + lr_min)
            if i > warmup_steps
            else (i / max(warmup_steps, 1) + lr_min)
        )
        for i in range(total_steps)
    }
    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[min(step, total_steps - 1)]
    )

    best_test_auc = 0.0
    best_epoch = -1
    if can_resume_training_state(train_state):
        if "optimizer" in train_state:
            try:
                optimizer.load_state_dict(train_state["optimizer"])
                print("Restored optimizer state.")
            except Exception as exc:
                print(f"Could not restore optimizer state: {exc}")
        if "scheduler" in train_state:
            try:
                scheduler.load_state_dict(train_state["scheduler"])
                print("Restored scheduler state.")
            except Exception as exc:
                print(f"Could not restore scheduler state: {exc}")
        if "scaler" in train_state:
            try:
                scaler.load_state_dict(train_state["scaler"])
                print("Restored AMP scaler state.")
            except Exception as exc:
                print(f"Could not restore AMP scaler state: {exc}")
        start_epoch = int(train_state.get("epoch", -1)) + 1
        best_test_auc = float(train_state.get("best_test_auc", 0.0))
        best_epoch = int(train_state.get("best_epoch", -1))
        print(f"Continuing training from epoch {start_epoch + 1}/{epochs}.")
    elif train_state:
        print("Loaded model weights from a non-mixed-curvature checkpoint; starting optimizer/scheduler from epoch 1.")

    for epoch in range(start_epoch, epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)
        model.train()
        train_labels, train_probs = [], []
        iter_i = epoch * iter_per_epoch

        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs = augment_batch(imgs)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, components, _ = model(imgs)
                loss, distortion, proto_ce = mixed_curvature_loss(
                    model._orig_mod if hasattr(model, "_orig_mod") else model,
                    logits,
                    components,
                    labels,
                    same_label_distance=args.same_label_distance,
                    different_label_distance=args.different_label_distance,
                    prototype_weight=args.prototype_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 256 == 0:
                live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
                print(
                    f"  batch={batch_idx:4d}/{iter_per_epoch}  "
                    f"L={loss.item():.4f}  L_dist={distortion.item():.4f}  "
                    f"CE={proto_ce.item():.4f}  {live_module.mixed_head.radius_summary()}"
                )

        print()
        compute_metrics(train_labels, train_probs, "Train", epoch)
        val_labels, val_probs = run_eval(model, val_loader, f"Epoch {epoch+1} [val]")
        val_auc = compute_metrics(val_labels, val_probs, "Val  ", epoch)

        if cdf_loader is not None:
            cdf_labels, cdf_probs = run_eval(model, cdf_loader, f"Epoch {epoch+1} [CDFv1]")
            test_auc = compute_metrics(cdf_labels, cdf_probs, "Test ", epoch)
            score_name = "Test AUC"
        else:
            test_auc = val_auc
            score_name = "Val AUC"

        live_module = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = live_module.state_dict()
        resume_state = make_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_test_auc,
            best_epoch,
        )

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            resume_state["best_test_auc"] = best_test_auc
            resume_state["best_epoch"] = best_epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            torch.save(resume_state, os.path.join(args.save_root, "best_resume.pth"))
            live_module.vit.save_pretrained(os.path.join(args.save_root, "best_lora"))
            print(f"\n  New best {score_name}={best_test_auc:.4f} -> saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  {score_name}={best_test_auc:.4f}")

        resume_state["best_test_auc"] = best_test_auc
        resume_state["best_epoch"] = best_epoch
        torch.save(state_dict, os.path.join(args.save_root, "latest.pth"))
        torch.save(resume_state, os.path.join(args.save_root, "latest_resume.pth"))
        live_module.vit.save_pretrained(os.path.join(args.save_root, "latest_lora"))

    print(f"\n{SEP}")
    print(f"  Training complete. Best checkpoint: epoch {best_epoch+1}  AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
