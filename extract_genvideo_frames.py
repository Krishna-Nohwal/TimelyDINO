"""
Uniformly extract frames from GenVideo videos into the repo's frame-manifest layout.

Input layout example:
    genvideo/
      OpenSora/train_OpenSora/*.mp4
      ZeroScope/train_ZeroScope/*.mp4

Output layout example:
    genvideo_16f/
      fake/opensora/train_OpenSora/OpenSora_10000_frame_00/image.png
      fake/zeroscope/train_ZeroScope/ZeroScope_00001_frame_00/image.png
      manifest_genvideo_16f.csv
      manifest_genvideo_16f_videos.csv

All GenVideo samples are fake:
    label = 1
"""

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
LABEL = 1
LABEL_NAME = "fake"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract uniform frames from all GenVideo videos."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        type=str,
        help="Path to GenVideo root, e.g. /media/.../usr/genvideo",
    )
    parser.add_argument(
        "--output_root",
        default="genvideo_16f",
        type=str,
        help="Directory where extracted frame folders and manifests are written.",
    )
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--max_opensora_videos", default=50, type=int)
    parser.add_argument("--max_zeroscope_videos", default=50, type=int)
    parser.add_argument("--sample_seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def natural_key(path: Path):
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def sanitize_part(text: str) -> str:
    text = text.strip().replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return text or "unnamed"


def normalize_source(name: str) -> str:
    compact = sanitize_part(name).lower().replace("_", "").replace("-", "")
    if "opensora" in compact:
        return "opensora"
    if "zeroscope" in compact:
        return "zeroscope"
    return sanitize_part(name).lower()


def safe_rel_parts(path: Path) -> list[str]:
    return [sanitize_part(part) for part in path.parts]


def is_relative_to(path: Path, maybe_parent: Path) -> bool:
    try:
        path.relative_to(maybe_parent)
        return True
    except ValueError:
        return False


def discover_videos(input_root: Path, output_root: Path) -> list[Path]:
    videos = []
    input_root_resolved = input_root.resolve()
    output_root_resolved = output_root.resolve()

    for dirpath, dirnames, filenames in os.walk(input_root_resolved, topdown=True, followlinks=False):
        current = Path(dirpath)
        kept = []
        for dirname in dirnames:
            child = current / dirname
            if child.is_symlink():
                continue
            if output_root_resolved.exists() and is_relative_to(child.resolve(), output_root_resolved):
                continue
            kept.append(dirname)
        dirnames[:] = kept

        for filename in filenames:
            path = current / filename
            if path.suffix.lower() in VIDEO_EXTS:
                videos.append(path)

    return sorted(videos, key=natural_key)


def subset_for_path(input_root: Path, video_path: Path) -> str:
    rel = video_path.relative_to(input_root)
    return normalize_source(rel.parts[0] if rel.parts else "unknown")


def limit_videos_by_subset(
    input_root: Path,
    videos: list[Path],
    max_opensora: int,
    max_zeroscope: int,
    seed: int,
) -> list[Path]:
    by_subset = {}
    for video in videos:
        source = subset_for_path(input_root, video)
        by_subset.setdefault(source, []).append(video)

    limits = {
        "opensora": max_opensora,
        "zeroscope": max_zeroscope,
    }
    rng = np.random.default_rng(seed)
    selected = []

    for source in sorted(by_subset):
        subset_videos = sorted(by_subset[source], key=natural_key)
        limit = limits.get(source, 0)
        if limit <= 0:
            chosen = subset_videos
            print(f"Subset limit {source}: using all {len(chosen)} videos")
        elif limit >= len(subset_videos):
            chosen = subset_videos
            print(f"Subset limit {source}: requested {limit}, available {len(chosen)}; using all")
        else:
            indices = sorted(rng.choice(len(subset_videos), size=limit, replace=False).tolist())
            chosen = [subset_videos[i] for i in indices]
            print(f"Subset limit {source}: selected {len(chosen)} / {len(subset_videos)} videos")
        selected.extend(chosen)

    return sorted(selected, key=natural_key)


def make_ids(input_root: Path, video_path: Path) -> tuple[str, str, Path]:
    rel_video = video_path.relative_to(input_root)
    rel_no_suffix = rel_video.with_suffix("")
    raw_parts = rel_no_suffix.parts
    source = normalize_source(raw_parts[0] if raw_parts else "unknown")
    safe_parts = safe_rel_parts(Path(*raw_parts[1:])) if len(raw_parts) > 1 else []
    video_id_path = Path(LABEL_NAME) / source
    for part in safe_parts:
        video_id_path = video_id_path / part
    return video_id_path.as_posix(), source, video_id_path


def read_frame_at(cap, frame_idx: int):
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame).convert("RGB")


def read_all_frames(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, Image.fromarray(frame).convert("RGB")))
        idx += 1
    cap.release()
    return frames


def save_frame(image: Image.Image, out_dir: Path, image_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = image.resize((image_size, image_size), Image.BICUBIC)
    image.save(out_dir / "image.png")


def extract_one_video(job: dict):
    import cv2

    input_root = Path(job["input_root"])
    output_root = Path(job["output_root"])
    video_path = Path(job["video_path"])
    num_frames = int(job["num_frames"])
    image_size = int(job["image_size"])
    overwrite = bool(job["overwrite"])

    video_id, source, video_id_path = make_ids(input_root, video_path)
    frame_rows = []
    video_row = {
        "video_id": video_id,
        "label": LABEL,
        "label_name": LABEL_NAME,
        "source": source,
        "orig_video": str(video_path),
        "num_requested_frames": num_frames,
        "num_extracted_frames": 0,
        "reported_total_frames": 0,
        "status": "pending",
        "note": "",
    }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture could not open video")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_row["reported_total_frames"] = total

        frames = []
        if total > 0:
            targets = np.linspace(0, max(total - 1, 0), num_frames)
            targets = np.round(targets).astype(int).tolist()
            for target in targets:
                image = read_frame_at(cap, target)
                if image is None:
                    frames = []
                    break
                frames.append((target, image))
        cap.release()

        if len(frames) != num_frames:
            all_frames = read_all_frames(video_path)
            if not all_frames:
                raise RuntimeError("no decodable frames")
            targets = np.linspace(0, len(all_frames) - 1, num_frames)
            targets = np.round(targets).astype(int).tolist()
            frames = [(all_frames[i][0], all_frames[i][1]) for i in targets]
            if total <= 0:
                video_row["reported_total_frames"] = len(all_frames)
            video_row["note"] = "used sequential fallback"

        for slot, (source_frame_idx, image) in enumerate(frames):
            sample_dir = (video_id_path.parent / f"{video_id_path.name}_frame_{slot:02d}").as_posix()
            frame_dir = output_root / sample_dir
            image_path = frame_dir / "image.png"
            if overwrite or not image_path.exists():
                save_frame(image, frame_dir, image_size)

            frame_rows.append({
                "sample_dir": sample_dir,
                "label": LABEL,
                "video_id": video_id,
                "label_name": LABEL_NAME,
                "source": source,
                "frame_slot": slot,
                "source_frame_idx": source_frame_idx,
                "orig_video": str(video_path),
                "image_path": str(image_path),
            })

        video_row["num_extracted_frames"] = len(frame_rows)
        video_row["status"] = "ok" if len(frame_rows) == num_frames else "partial"
        return {"ok": True, "frame_rows": frame_rows, "video_row": video_row}

    except Exception as exc:
        video_row["status"] = "failed"
        video_row["note"] = repr(exc)
        return {"ok": False, "frame_rows": [], "video_row": video_row}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_discovery_summary(input_root: Path, videos: list[Path]):
    counts = {}
    for path in videos:
        source = subset_for_path(input_root, path)
        counts[source] = counts.get(source, 0) + 1

    print(f"Input root: {input_root}")
    print(f"Discovered GenVideo videos: {len(videos)}")
    print("By subset:")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    discovered_videos = discover_videos(input_root, output_root)
    print("Before subset limiting:")
    print_discovery_summary(input_root, discovered_videos)

    videos = limit_videos_by_subset(
        input_root,
        discovered_videos,
        args.max_opensora_videos,
        args.max_zeroscope_videos,
        args.sample_seed,
    )
    print("\nAfter subset limiting:")
    print_discovery_summary(input_root, videos)

    if args.dry_run:
        print("Dry run enabled; no frames extracted.")
        return
    if not videos:
        raise RuntimeError("No videos found.")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Overwrite existing frames: {args.overwrite}")
    print("Labels: all GenVideo samples are fake (label=1)")

    jobs = [
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "video_path": str(video),
            "num_frames": args.num_frames,
            "image_size": args.image_size,
            "overwrite": args.overwrite,
        }
        for video in videos
    ]

    frame_rows = []
    video_rows = []
    if args.num_workers <= 1:
        iterator = (extract_one_video(job) for job in jobs)
        for result in tqdm(iterator, total=len(jobs), desc="Extracting GenVideo"):
            frame_rows.extend(result["frame_rows"])
            video_rows.append(result["video_row"])
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(extract_one_video, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting GenVideo"):
                result = future.result()
                frame_rows.extend(result["frame_rows"])
                video_rows.append(result["video_row"])

    frame_rows = sorted(frame_rows, key=lambda row: natural_key(Path(row["sample_dir"])))
    video_rows = sorted(video_rows, key=lambda row: natural_key(Path(row["video_id"])))

    manifest_path = output_root / f"manifest_genvideo_{args.num_frames}f.csv"
    videos_manifest_path = output_root / f"manifest_genvideo_{args.num_frames}f_videos.csv"

    write_csv(
        manifest_path,
        frame_rows,
        [
            "sample_dir",
            "label",
            "video_id",
            "label_name",
            "source",
            "frame_slot",
            "source_frame_idx",
            "orig_video",
            "image_path",
        ],
    )
    write_csv(
        videos_manifest_path,
        video_rows,
        [
            "video_id",
            "label",
            "label_name",
            "source",
            "orig_video",
            "num_requested_frames",
            "num_extracted_frames",
            "reported_total_frames",
            "status",
            "note",
        ],
    )

    ok_videos = sum(row["status"] == "ok" for row in video_rows)
    failed_videos = sum(row["status"] == "failed" for row in video_rows)
    partial_videos = sum(row["status"] == "partial" for row in video_rows)

    print("\nDone.")
    print(f"  extracted frame rows: {len(frame_rows)}")
    print(f"  ok videos: {ok_videos}")
    print(f"  partial videos: {partial_videos}")
    print(f"  failed videos: {failed_videos}")
    print(f"  frame manifest: {manifest_path}")
    print(f"  video manifest: {videos_manifest_path}")
    if failed_videos:
        print("\nFailed videos:")
        for row in video_rows:
            if row["status"] == "failed":
                print(f"  {row['orig_video']}  ({row['note']})")


if __name__ == "__main__":
    main()
