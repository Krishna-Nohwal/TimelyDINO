"""
test_cdfv2_no_memory_stage2.py

Evaluate a no-memory Stage-2 checkpoint on CDFv2.

Expected layout:
    <fake_root>/<sample_dir>/image.png
    <real_root>/<sample_dir>/image.png

Example:
python test_cdfv2_no_memory_stage2.py \
    --checkpoint checkpoints_s2_frame_end_no_memory/best.pth \
    --fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --num_frames 32 \
    --batch_size 4
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from augmentations import load_and_resize, normalize
from stage2_no_memory_eval_common import (
    IMG_SIZE,
    evaluate_dataset,
    load_clip_frames,
    sample_frame_indices,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate no-memory Stage-2 VideoViT on CDFv2."
    )
    p.add_argument("--checkpoint", default="checkpoints_s2_frame_end_no_memory/best.pth")
    p.add_argument("--fake_root",
                   default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2")
    p.add_argument("--real_root",
                   default="/media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real")
    p.add_argument("--num_frames", default=32, type=int)
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--topk", default=10, type=int)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--real_bias", default=0.0, type=float)
    p.add_argument("--save_results", default="")
    return p.parse_args()


def video_id_from_sample(sample_name: str) -> str:
    return re.sub(r"_(?:frame_|f)\d+$", "", sample_name)


def collect_samples(root: Path, label: int):
    samples = []
    if not root.is_dir():
        raise FileNotFoundError(f"Root does not exist: {root}")
    for sample_dir in sorted(root.iterdir()):
        if sample_dir.is_dir() and (sample_dir / "image.png").is_file():
            samples.append((sample_dir, label))
    return samples


class CDFv2FrameDataset(Dataset):
    def __init__(self, fake_root: Path, real_root: Path):
        fake_samples = collect_samples(fake_root, label=1)
        real_samples = collect_samples(real_root, label=0)
        self.samples = fake_samples + real_samples
        print(
            f"  CDFv2 frames -> Real: {len(real_samples)} | "
            f"Fake: {len(fake_samples)} | Total: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]
        img = load_and_resize(str(sample_dir / "image.png"), IMG_SIZE)
        img = normalize(img)
        video_id = video_id_from_sample(sample_dir.name)
        return img, label, video_id


class CDFv2ClipDataset(Dataset):
    def __init__(self, fake_root: Path, real_root: Path, num_frames: int):
        self.num_frames = num_frames
        vid2paths = defaultdict(list)
        vid2label = {}

        for root, label in [(fake_root, 1), (real_root, 0)]:
            if not root.is_dir():
                raise FileNotFoundError(f"Root does not exist: {root}")
            for sample_dir in sorted(root.iterdir()):
                img_path = sample_dir / "image.png"
                if sample_dir.is_dir() and img_path.is_file():
                    vid = video_id_from_sample(sample_dir.name)
                    vid2paths[vid].append(str(img_path))
                    vid2label[vid] = label

        self.videos = [
            (vid, sorted(paths), vid2label[vid])
            for vid, paths in sorted(vid2paths.items())
        ]
        real_n = sum(1 for _, _, label in self.videos if label == 0)
        fake_n = sum(1 for _, _, label in self.videos if label == 1)
        print(f"  CDFv2 clips  -> Real: {real_n} | Fake: {fake_n} | Total: {len(self.videos)}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, paths, label = self.videos[idx]
        indices = sample_frame_indices(len(paths), self.num_frames)
        frames = load_clip_frames(paths, indices)
        return frames, label, vid


def main():
    args = parse_args()
    fake_root = Path(args.fake_root)
    real_root = Path(args.real_root)
    print(f"  Fake root: {fake_root}")
    print(f"  Real root: {real_root}")

    frame_dataset = CDFv2FrameDataset(fake_root, real_root)
    clip_dataset = CDFv2ClipDataset(fake_root, real_root, args.num_frames)
    evaluate_dataset(
        dataset_name="CDFv2",
        checkpoint=args.checkpoint,
        frame_dataset=frame_dataset,
        clip_dataset=clip_dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        topk=args.topk,
        no_compile=args.no_compile,
        fp32=args.fp32,
        real_bias=args.real_bias,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
