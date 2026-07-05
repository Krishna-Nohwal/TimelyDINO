"""
test_wdf_no_memory_stage2.py

Evaluate a no-memory Stage-2 checkpoint on WDF.

Expected layout:
    <fake_root>/<video_id>_<frame_number>.png
    <real_root>/<video_id>_<frame_number>.png

Example:
python test_wdf_no_memory_stage2.py \
    --checkpoint checkpoints_s2_frame_end_no_memory/best.pth \
    --fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --num_frames 32 \
    --batch_size 4
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from augmentations import load_and_resize, normalize
from stage2_no_memory_eval_common import (
    IMG_SIZE,
    evaluate_dataset,
    load_clip_frames,
    sample_frame_indices,
)


FRAME_FILE_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate no-memory Stage-2 VideoViT on WDF."
    )
    p.add_argument("--checkpoint", default="checkpoints_s2_frame_end_no_memory/best.pth")
    p.add_argument("--fake_root",
                   default="/media/tarun/B482367C823642E2/usr/wdf/test/fake")
    p.add_argument("--real_root",
                   default="/media/tarun/B482367C823642E2/usr/wdf/test/real")
    p.add_argument("--num_frames", default=32, type=int)
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--topk", default=10, type=int)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--real_bias", default=0.0, type=float)
    p.add_argument("--save_results", default="")
    return p.parse_args()


def group_wdf_videos(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"Root does not exist: {root}")

    grouped = defaultdict(list)
    skipped = 0
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_FILE_RE.match(path.name)
        if not match:
            skipped += 1
            continue
        video_id = match.group(1)
        frame_idx = int(match.group(2))
        grouped[video_id].append((frame_idx, str(path)))

    if skipped:
        print(f"  [WDF] skipped {skipped} files under {root}")

    videos = {}
    for video_id, items in sorted(grouped.items()):
        videos[video_id] = [path for _, path in sorted(items)]
    return videos


class WDFFrameDataset(Dataset):
    def __init__(self, fake_root: Path, real_root: Path):
        self.samples = []
        for root, label in [(fake_root, 1), (real_root, 0)]:
            grouped = group_wdf_videos(root)
            for video_id, paths in grouped.items():
                for path in paths:
                    self.samples.append((path, label, video_id))

        real_n = sum(1 for _, label, _ in self.samples if label == 0)
        fake_n = sum(1 for _, label, _ in self.samples if label == 1)
        print(f"  WDF frames -> Real: {real_n} | Fake: {fake_n} | Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, video_id = self.samples[idx]
        img = load_and_resize(path, IMG_SIZE)
        img = normalize(img)
        return img, label, video_id


class WDFClipDataset(Dataset):
    def __init__(self, fake_root: Path, real_root: Path, num_frames: int):
        self.num_frames = num_frames
        self.videos = []
        for root, label in [(fake_root, 1), (real_root, 0)]:
            grouped = group_wdf_videos(root)
            for video_id, paths in grouped.items():
                if paths:
                    self.videos.append((video_id, paths, label))

        real_n = sum(1 for _, _, label in self.videos if label == 0)
        fake_n = sum(1 for _, _, label in self.videos if label == 1)
        print(f"  WDF clips  -> Real: {real_n} | Fake: {fake_n} | Total: {len(self.videos)}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_id, paths, label = self.videos[idx]
        indices = sample_frame_indices(len(paths), self.num_frames)
        frames = load_clip_frames(paths, indices)
        return frames, label, video_id


def main():
    args = parse_args()
    fake_root = Path(args.fake_root)
    real_root = Path(args.real_root)
    print(f"  Fake root: {fake_root}")
    print(f"  Real root: {real_root}")

    frame_dataset = WDFFrameDataset(fake_root, real_root)
    clip_dataset = WDFClipDataset(fake_root, real_root, args.num_frames)
    evaluate_dataset(
        dataset_name="WDF",
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
