import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import load_and_resize, normalize
from frame_model import ViT
from video_model import TemporalTransformer


IMG_SIZE = 256
DEEPEST_HEAD_IDX = ViT.NUM_LAYERS - 1


class VideoViTNoMemory(torch.nn.Module):
    """Stage-2 frame-end VideoViT without any memory-bank/retrieval branch."""

    EMBED_DIM = ViT.EMBED_DIM
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_model = ViT()
        self.temporal_transformers = torch.nn.ModuleList([
            TemporalTransformer(
                embed_dim=self.EMBED_DIM,
                num_frames=num_frames,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])
        self.fusion_classifier = torch.nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM + 2,
            2,
        )

    @staticmethod
    def _mean_valid_frame_logits(frame_logits_list, bsz, time_steps, key_padding_mask, dtype):
        frame_logits = frame_logits_list[-1].float().reshape(bsz, time_steps, 2)
        if key_padding_mask is None:
            return frame_logits.mean(dim=1).to(dtype=dtype)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        counts = valid.sum(dim=1).clamp(min=1)
        return ((frame_logits * valid).sum(dim=1) / counts).to(dtype=dtype)

    def forward(self, video: torch.Tensor, lengths=None):
        bsz, time_steps, channels, height, width = video.shape
        if time_steps > self.num_frames:
            raise ValueError(f"Expected <= {self.num_frames} frames, got {time_steps}")

        frames = video.reshape(bsz * time_steps, channels, height, width)
        frame_logits_list, frame_feats_list, cls_list = self.frame_model(frames)

        if lengths is None:
            key_padding_mask = None
        else:
            time_idx = torch.arange(time_steps, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)

        cls_sequences = [
            cls_tokens.reshape(bsz, time_steps, self.EMBED_DIM) for cls_tokens in cls_list
        ]

        video_feats_list = []
        for temporal_tfm, frame_cls in zip(self.temporal_transformers, cls_sequences):
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)
        frame_mean_logits = self._mean_valid_frame_logits(
            frame_logits_list, bsz, time_steps, key_padding_mask, temporal_vec.dtype
        )

        fused_with_frame = torch.cat([temporal_vec, frame_mean_logits], dim=1)
        fused_no_frame = torch.cat([temporal_vec, torch.zeros_like(frame_mean_logits)], dim=1)

        video_logits_with_frame = self.fusion_classifier(fused_with_frame)
        video_logits_no_frame = self.fusion_classifier(fused_no_frame)

        return (
            video_logits_with_frame,
            video_logits_no_frame,
            frame_logits_list,
            frame_feats_list,
            video_feats_list,
        )


def load_no_memory_model(checkpoint_path: str, num_frames: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}

    if any("memory" in k or "gate" in k for k in ckpt):
        raise ValueError(
            "This checkpoint contains memory/gate parameters. Use a no-memory "
            "Stage-2 checkpoint for this evaluator."
        )

    fusion_key = "fusion_classifier.weight"
    if fusion_key not in ckpt:
        raise KeyError(f"Cannot find '{fusion_key}' in checkpoint.")
    expected_dim = VideoViTNoMemory.NUM_TEMPORAL_HEADS * VideoViTNoMemory.EMBED_DIM + 2
    actual_dim = int(ckpt[fusion_key].shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Unexpected fusion_classifier input dim {actual_dim}; expected "
            f"{expected_dim} for no-memory frame-end Stage 2."
        )

    model = VideoViTNoMemory(num_frames=num_frames).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys: {unexpected[:5]}")
    print("  No-memory Stage-2 checkpoint loaded successfully.")
    return model


def sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def load_clip_frames(paths: List[str], indices: np.ndarray) -> torch.Tensor:
    frames = []
    for idx in indices:
        try:
            img = load_and_resize(paths[int(idx)], IMG_SIZE)
            img = normalize(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        frames.append(img)
    return torch.stack(frames, dim=0)


def clip_collate_fn(batch):
    frames_list, labels, vids = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for frames in frames_list:
        pad_t = max_len - frames.size(0)
        if pad_t > 0:
            frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded.append(frames)
    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
        list(vids),
    )


def compute_and_print_metrics(labels, probs, level: str) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    fpr_arr, tpr_arr, _ = roc_curve(labels, probs, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sep = "-" * 72
    print(f"\n  {sep}")
    print(f"  [{level}]")
    print(f"  {sep}")
    print(f"  AUC  : {auc:.4f}")
    print(f"  AP   : {ap:.4f}")
    print(f"  Acc  : {acc * 100:.2f}%")
    print(f"  F1   : {f1:.4f}")
    print(f"  EER  : {eer * 100:.2f}%")
    print(f"  TPR  : {tpr * 100:.2f}%   FPR : {fpr * 100:.2f}%   TNR : {tnr * 100:.2f}%")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  {sep}")
    return float(auc)


def apply_real_bias(probs: list, real_bias: float) -> list:
    if real_bias == 0.0:
        return probs
    exponent = 1.0 + real_bias
    return [p ** exponent if p < 0.5 else p for p in probs]


def run_frame_inference(model, loader, device, use_fp32=False, topk=10):
    frame_labels, frame_probs, frame_logits, frame_vids = [], [], [], []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not use_fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    frame_model = model._orig_mod.frame_model if hasattr(model, "_orig_mod") else model.frame_model
    frame_model.eval()

    with torch.inference_mode(), autocast_ctx:
        for imgs, labels, video_ids in tqdm(loader, desc="Frame inference", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            logits_list, _, _ = frame_model(imgs)
            raw_logits = logits_list[DEEPEST_HEAD_IDX].float()
            probs = torch.softmax(raw_logits, dim=1)[:, 1].cpu().numpy()
            fake_logit = raw_logits[:, 1].cpu().numpy()

            frame_probs.extend(probs.tolist())
            frame_logits.extend(fake_logit.tolist())
            frame_labels.extend(labels.numpy().tolist())
            frame_vids.extend(list(video_ids))

    vid2labels = defaultdict(list)
    vid2probs = defaultdict(list)
    vid2logits = defaultdict(list)
    for label, prob, logit, vid in zip(frame_labels, frame_probs, frame_logits, frame_vids):
        vid2labels[vid].append(label)
        vid2probs[vid].append(prob)
        vid2logits[vid].append(logit)

    video_ids = sorted(vid2labels.keys())
    video_labels, mean_probs, mean_logits, topk_mean = [], [], [], []

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for vid in video_ids:
        labels = vid2labels[vid]
        if len(set(labels)) > 1:
            print(f"  [WARNING] Video '{vid}' has mixed labels {set(labels)}; using majority.")
        video_labels.append(int(round(np.mean(labels))))

        frame_p = np.asarray(vid2probs[vid])
        frame_l = np.asarray(vid2logits[vid])
        mean_probs.append(float(frame_p.mean()))
        mean_logits.append(float(sigmoid(frame_l.mean())))
        k = min(topk, len(frame_p))
        topk_mean.append(float(np.partition(frame_p, -k)[-k:].mean()))

    return frame_labels, frame_probs, video_ids, video_labels, mean_probs, mean_logits, topk_mean


def run_clip_inference(model, loader, device, use_fp32=False):
    video_ids, video_labels = [], []
    frame_end_probs, no_frame_probs = [], []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if not use_fp32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    model.eval()
    with torch.inference_mode(), autocast_ctx:
        for frames, labels, lengths, vids in tqdm(loader, desc="Clip inference", unit="batch"):
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            video_logits_with_frame, video_logits_no_frame, _, _, _ = model(frames, lengths)
            frame_end = torch.softmax(video_logits_with_frame.float(), dim=1)[:, 1]
            no_frame = torch.softmax(video_logits_no_frame.float(), dim=1)[:, 1]

            frame_end_probs.extend(frame_end.cpu().numpy().tolist())
            no_frame_probs.extend(no_frame.cpu().numpy().tolist())
            video_labels.extend(labels.numpy().tolist())
            video_ids.extend(vids)

    return video_ids, video_labels, frame_end_probs, no_frame_probs


def evaluate_dataset(
    dataset_name: str,
    checkpoint: str,
    frame_dataset,
    clip_dataset,
    batch_size: int,
    num_frames: int,
    num_workers: int,
    topk: int,
    no_compile: bool,
    fp32: bool,
    real_bias: float,
    save_results: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    sep = "=" * 72
    print(f"\n  {sep}")
    print(f"  {dataset_name} EVALUATION -- no-memory Stage 2")
    print(f"  {sep}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Num frames : {num_frames}")
    print(f"  Batch size : {batch_size}")
    print(f"  Precision  : {'FP32' if fp32 else 'FP16 autocast'}")

    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    frame_loader = DataLoader(
        frame_dataset,
        batch_size=batch_size * num_frames,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    clip_loader = DataLoader(
        clip_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    model = load_no_memory_model(checkpoint, num_frames, device)
    if not no_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile ...")
        model = torch.compile(model)

    print("\n  Running frame-level inference ...")
    (
        frame_labels,
        frame_probs,
        vid_ids_frame,
        vid_labels_frame,
        vid_mean_probs,
        vid_mean_logits,
        vid_topk_mean,
    ) = run_frame_inference(model, frame_loader, device, use_fp32=fp32, topk=topk)

    print("\n  Running clip-level temporal inference ...")
    vid_ids_clip, vid_labels_clip, vid_frame_end_probs, vid_no_frame_probs = run_clip_inference(
        model, clip_loader, device, use_fp32=fp32
    )

    bias_tag = ""
    if real_bias != 0.0:
        bias_tag = f" [real_bias={real_bias}]"
        vid_mean_probs = apply_real_bias(vid_mean_probs, real_bias)
        vid_mean_logits = apply_real_bias(vid_mean_logits, real_bias)
        vid_topk_mean = apply_real_bias(vid_topk_mean, real_bias)
        vid_frame_end_probs = apply_real_bias(vid_frame_end_probs, real_bias)
        vid_no_frame_probs = apply_real_bias(vid_no_frame_probs, real_bias)

    auc_frame = compute_and_print_metrics(
        frame_labels, frame_probs, f"(A) Frame-level{bias_tag} ({dataset_name})"
    )
    auc_mean_probs = compute_and_print_metrics(
        vid_labels_frame, vid_mean_probs, f"(B) Video-mean probs{bias_tag} ({dataset_name})"
    )
    auc_mean_logits = compute_and_print_metrics(
        vid_labels_frame,
        vid_mean_logits,
        f"(B) Video-mean logits->sigmoid{bias_tag} ({dataset_name})",
    )
    auc_topk = compute_and_print_metrics(
        vid_labels_frame,
        vid_topk_mean,
        f"(B) Video-top{topk}-mean probs{bias_tag} ({dataset_name})",
    )
    auc_frame_end = compute_and_print_metrics(
        vid_labels_clip,
        vid_frame_end_probs,
        f"(C) Video-temporal frame-end{bias_tag} ({dataset_name})",
    )
    auc_no_frame = compute_and_print_metrics(
        vid_labels_clip,
        vid_no_frame_probs,
        f"(D) Video-temporal no-frame{bias_tag} ({dataset_name})",
    )

    if save_results:
        frame_side = {
            vid: {
                "label": label,
                "mean_prob": mean_prob,
                "mean_logit_prob": mean_logit,
                f"top{topk}_mean_prob": topk_prob,
            }
            for vid, label, mean_prob, mean_logit, topk_prob in zip(
                vid_ids_frame, vid_labels_frame, vid_mean_probs, vid_mean_logits, vid_topk_mean
            )
        }
        clip_frame_end = dict(zip(vid_ids_clip, vid_frame_end_probs))
        clip_no_frame = dict(zip(vid_ids_clip, vid_no_frame_probs))
        rows = []
        for vid in sorted(frame_side.keys()):
            row = {"video_id": vid}
            row.update(frame_side[vid])
            row["frame_end_prob"] = clip_frame_end.get(vid, float("nan"))
            row["no_frame_prob"] = clip_no_frame.get(vid, float("nan"))
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_results, index=False)
        print(f"\n  Per-video results written to: {save_results}")

    print(f"\n  {sep}")
    print(f"  FINAL SUMMARY [{dataset_name} no-memory]{bias_tag}")
    print(f"  {sep}")
    print(f"  (A) Frame-level AUC                    : {auc_frame:.4f}")
    print(f"  (B) Video-mean probs AUC               : {auc_mean_probs:.4f}")
    print(f"  (B) Video-mean logits->sigmoid AUC     : {auc_mean_logits:.4f}")
    print(f"  (B) Video-top{topk}-mean probs AUC      : {auc_topk:.4f}")
    print(f"  (C) Video-temporal frame-end AUC       : {auc_frame_end:.4f}  <- primary")
    print(f"  (D) Video-temporal no-frame AUC        : {auc_no_frame:.4f}")
    print(f"  {sep}")
