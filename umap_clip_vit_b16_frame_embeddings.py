"""
UMAP real/fake split for a trained CLIP ViT-B/16 frame model, aggregated to videos.

This loads checkpoints produced by train_clip_vit_b16_all.py, extracts a
penultimate CLIP ViT frame embedding, mean-pools embeddings per video, and saves
only a real-vs-fake split UMAP.

Example:
python umap_clip_vit_b16_frame_embeddings.py \
    --checkpoint checkpoints_clip_vit_b16_all/best.pth \
    --manifest /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out/manifest_ff_onct.csv \
    --root_dir /media/tarun/B482367C823642E2/usr/ff++/onct_preprocessed_out \
    --cdfv2_fake_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/fake/cdfv2 \
    --cdfv2_real_root /media/tarun/B482367C823642E2/usr/preprocessed_cdfv2_test32/real \
    --cdfv3_root /media/tarun/B482367C823642E2/usr/cdfv3_face_crops \
    --df0_fake_root /media/tarun/B482367C823642E2/usr/df1.0_faces/fake \
    --df0_real_root /media/tarun/B482367C823642E2/usr/df1.0_faces/real \
    --dfd_fake_root /media/tarun/B482367C823642E2/usr/dfd_faces/fake \
    --dfd_real_root /media/tarun/B482367C823642E2/usr/dfd_faces/real \
    --dfdc_fake_root /media/tarun/B482367C823642E2/usr/dfdc/fake \
    --dfdc_real_root /media/tarun/B482367C823642E2/usr/dfdc/real \
    --wdf_fake_root /media/tarun/B482367C823642E2/usr/wdf/test/fake \
    --wdf_real_root /media/tarun/B482367C823642E2/usr/wdf/test/real \
    --uadfv_fake_root /media/tarun/B482367C823642E2/usr/uadfv_faces/fake \
    --uadfv_real_root /media/tarun/B482367C823642E2/usr/uadfv_faces/real \
    --frames_per_video 4 \
    --max_frames_per_dataset 200 \
    --out umap_clip_vit_b16_real_fake_split.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from umap_xception_frame_embeddings import (
    FrameDataset,
    aggregate_frames_to_videos,
    build_frame_items,
    default_embeddings_path,
    load_cached,
    patch_coverage_for_numba,
    plot_real_fake_split,
    print_frame_counts,
    print_sampled_auc,
    remove_outliers,
)


def parse_args():
    p = argparse.ArgumentParser(description="Real/fake split UMAP for CLIP ViT-B/16 frame embeddings.")
    p.add_argument("--checkpoint", default="", help="CLIP ViT-B/16 checkpoint from train_clip_vit_b16_all.py.")
    p.add_argument("--embeddings_npz", default="", help="Cached embeddings from this script.")
    p.add_argument("--embeddings_out", default="", help="Cache output path. Default: next to --out.")
    p.add_argument("--out", default="umap_clip_vit_b16_real_fake_split.png")
    p.add_argument("--model_name", default="", help="Override checkpoint model_name. Usually leave empty.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--frames_per_video", type=int, default=4, help="0 = use all frames from selected videos.")
    p.add_argument("--max_videos_per_dataset", type=int, default=0, help="0 = no video cap.")
    p.add_argument("--max_frames_per_dataset", type=int, default=0, help="0 = no per-dataset frame cap.")
    p.add_argument("--max_total_frames", type=int, default=0, help="0 = no global frame cap after dataset sampling.")
    p.add_argument("--val_ratio", type=float, default=0.0, help="FF++ val split ratio. 0 = all FF++ videos.")

    # FF++
    p.add_argument("--manifest", default="")
    p.add_argument("--root_dir", default="")

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # CDFv3 / CDF++
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # DFo / DeeperForensics-1.0
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")

    # DFD
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")

    # WDF
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")

    # UADFV
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")

    # UMAP
    p.add_argument("--outlier_std", type=float, default=2.0, help="<=0 disables outlier removal.")
    p.add_argument("--outlier_group", default="class", choices=["class", "dataset_class", "dataset", "all"])
    p.add_argument("--umap_neighbors", type=int, default=50)
    p.add_argument("--umap_min_dist", type=float, default=0.02)
    p.add_argument("--umap_metric", default="cosine")
    p.add_argument("--umap_seed", type=int, default=42)
    return p.parse_args()


def build_clip_vit(model_name: str):
    import timm

    return timm.create_model(model_name, pretrained=False, num_classes=2)


def clean_state_dict(obj):
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj))) if isinstance(obj, dict) else obj
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_model(checkpoint: str, model_name_override: str, device: torch.device):
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_name = raw.get("model_name", "vit_base_patch16_clip_224.openai") if isinstance(raw, dict) else "vit_base_patch16_clip_224.openai"
    model_name = model_name_override or checkpoint_model_name
    model = build_clip_vit(model_name)
    state = clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"  model_name     : {model_name}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    return model.to(device).eval(), model_name


def _feature_tensor_from_forward_features(model, features):
    if isinstance(features, dict):
        for key in ("pre_logits", "pooled", "x_norm_clstoken", "cls_token"):
            if key in features and features[key] is not None:
                value = features[key]
                return value[:, 0] if value.dim() == 3 else value
        for key in ("x", "tokens"):
            if key in features and features[key] is not None:
                value = features[key]
                return value[:, 0] if value.dim() == 3 else value.flatten(1)
        raise RuntimeError(f"Unknown forward_features dict keys: {list(features.keys())}")

    if features.dim() == 3:
        return features[:, 0]
    if features.dim() == 4:
        return features.mean(dim=(2, 3))
    return features.flatten(1)


def forward_features_and_logits(model, images: torch.Tensor):
    features = model.forward_features(images)
    emb = None
    logits = None

    if hasattr(model, "forward_head"):
        try:
            emb = model.forward_head(features, pre_logits=True)
        except (TypeError, RuntimeError, AttributeError):
            emb = None
        try:
            logits = model.forward_head(features)
        except (TypeError, RuntimeError, AttributeError):
            logits = None

    if emb is None:
        emb = _feature_tensor_from_forward_features(model, features)
    if emb.dim() > 2:
        emb = emb.flatten(1)

    if logits is None:
        classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
        if classifier is not None and not isinstance(classifier, torch.nn.Identity):
            logits = classifier(emb)
        else:
            logits = model(images)
    return emb.float(), logits.float()


@torch.inference_mode()
def extract_embeddings(model, loader, device: torch.device, args):
    embeddings, labels, probs = [], [], []
    paths, datasets, video_ids, frame_positions, ok_flags = [], [], [], [], []
    amp_enabled = device.type == "cuda" and not args.no_amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)

    with autocast:
        for images, batch_labels, batch_paths, batch_dsets, batch_vids, batch_pos, batch_ok in DataLoaderProgress(loader):
            images = images.to(device, non_blocking=True)
            emb, logits = forward_features_and_logits(model, images)
            prob = torch.softmax(logits, dim=1)[:, 1]
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probs.extend(prob.cpu().numpy().tolist())
            paths.extend(list(batch_paths))
            datasets.extend(list(batch_dsets))
            video_ids.extend(list(batch_vids))
            frame_positions.extend(batch_pos.numpy().astype(int).tolist())
            ok_flags.extend(batch_ok.numpy().astype(bool).tolist())

    return (
        np.concatenate(embeddings, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(probs, dtype=np.float32),
        paths,
        datasets,
        video_ids,
        np.asarray(frame_positions, dtype=np.int64),
        np.asarray(ok_flags, dtype=bool),
    )


def DataLoaderProgress(loader):
    from tqdm import tqdm

    return tqdm(loader, desc="Extracting CLIP ViT-B/16 features")


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("=" * 88)
    print("UMAP real/fake split for CLIP ViT-B/16 frame embeddings")
    print("=" * 88)
    print(f"Device : {device}")
    print(f"Output : {args.out}")

    if args.embeddings_npz:
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = load_cached(args.embeddings_npz)
        print(f"Loaded cached embeddings: {embeddings.shape} from {args.embeddings_npz}")
    else:
        if not args.checkpoint:
            raise ValueError("Supply --checkpoint or --embeddings_npz.")
        items = build_frame_items(args)
        print(f"\nTotal sampled frames: {len(items)}")
        loader = DataLoader(
            FrameDataset(items, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        model, model_name = load_model(args.checkpoint, args.model_name, device)
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = extract_embeddings(
            model, loader, device, args
        )
        print(f"Extracted embeddings: {embeddings.shape}")
        print(f"Bad/blank image fallbacks: {int((~ok_flags).sum())}")

        cache_path = args.embeddings_out or default_embeddings_path(args.out)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            labels=labels,
            probs=probs,
            paths=np.asarray(paths),
            dataset_tags=np.asarray(dataset_tags),
            video_ids=np.asarray(video_ids),
            frame_positions=frame_positions,
            ok_flags=ok_flags,
            model_name=np.asarray([model_name]),
        )
        print(f"Cached embeddings -> {cache_path}")

    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = aggregate_frames_to_videos(
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags
    )
    print_frame_counts("Videos before outlier removal:", labels, dataset_tags)
    print_sampled_auc("Video-level performance from mean frame probabilities:", labels, probs, dataset_tags)
    embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags = remove_outliers(
        embeddings, labels, probs, paths, dataset_tags, video_ids, frame_positions, ok_flags,
        args.outlier_std, args.outlier_group,
    )
    print_frame_counts("Videos used for UMAP:", labels, dataset_tags)
    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 videos for UMAP after filtering.")

    from sklearn.preprocessing import StandardScaler
    patch_coverage_for_numba()
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install UMAP with: pip install umap-learn") from exc
    except AttributeError as exc:
        raise RuntimeError(
            "UMAP import failed inside numba/coverage. Embeddings are cached, so rerun "
            "with --embeddings_npz <cache>, or run: pip install -U numba coverage umap-learn"
        ) from exc

    print("\nStandardizing embeddings and fitting UMAP ...")
    scaled = StandardScaler().fit_transform(embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(args.umap_neighbors, max(2, scaled.shape[0] - 1)),
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
    )
    coords = reducer.fit_transform(scaled)

    plot_real_fake_split(coords, labels, dataset_tags, args.out)
    out_path = Path(args.out)
    coords_path = str(out_path.with_name(f"{out_path.stem}_coords.csv"))
    pd.DataFrame({
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "label": labels,
        "prob_fake": probs,
        "dataset": dataset_tags,
        "video_id": video_ids,
        "n_frames": frame_positions,
        "example_path": paths,
    }).to_csv(coords_path, index=False)
    print("\nSaved outputs:")
    print(f"  {args.out}")
    print(f"  {coords_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
