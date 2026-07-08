"""
Train CLIP ViT-B/16 on FF++ plus all evaluation-style datasets.

This uses the shared dataset handling from train_xception_ffpp.py:
    FF++, CDFv2, CDF++/CDFv3, UADFV, DFo, DFD, DFDC, and WDF.

Default backbone:
    timm: vit_base_patch16_clip_224.openai

Example:
python train_clip_vit_b16_all.py \
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
    --save_dir checkpoints_clip_vit_b16_all \
    --frames_per_video 8 \
    --epochs 10 \
    --batch_size 64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_xception_ffpp import (
    FrameDataset,
    cap_df,
    evaluate,
    make_class_weight,
    prepare_video_split,
    train_one_epoch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP ViT-B/16 on all combined datasets.")
    p.add_argument("--manifest", required=True, help="FF++ manifest CSV with sample_dir,label.")
    p.add_argument("--root_dir", required=True, help="FF++ preprocessed frame root.")
    p.add_argument("--save_dir", default="checkpoints_clip_vit_b16_all")
    p.add_argument(
        "--model_name",
        default="vit_base_patch16_clip_224.openai",
        help="timm CLIP ViT-B/16 model name.",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames_per_video", type=int, default=0, help="0 = use all frames from every selected video.")
    p.add_argument("--max_train_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--max_val_frames", type=int, default=0, help="0 = no cap.")
    p.add_argument("--no_amp", action="store_true")

    # CDFv2
    p.add_argument("--cdfv2_fake_root", default="")
    p.add_argument("--cdfv2_real_root", default="")

    # CDFv3 / CDF++ manifest layout. Manifest labels: 1=Real, 0=Fake.
    p.add_argument("--cdfv3_root", default="")
    p.add_argument("--cdfv3_csv", default="")

    # UADFV / DFo / DFD nested layouts: <root>/<video>/<frame>/image.png
    p.add_argument("--uadfv_fake_root", default="")
    p.add_argument("--uadfv_real_root", default="")
    p.add_argument("--df0_fake_root", default="")
    p.add_argument("--df0_real_root", default="")
    p.add_argument("--dfo_fake_root", default="")
    p.add_argument("--dfo_real_root", default="")
    p.add_argument("--dfd_fake_root", default="")
    p.add_argument("--dfd_real_root", default="")

    # DFDC / WDF flat layouts: <root>/<video_id>_<frame>.png
    p.add_argument("--dfdc_fake_root", default="")
    p.add_argument("--dfdc_real_root", default="")
    p.add_argument("--wdf_fake_root", default="")
    p.add_argument("--wdf_real_root", default="")
    return p.parse_args()


def build_clip_vit(model_name: str):
    import timm

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=2)
    except Exception as exc:
        msg = (
            f"Could not create timm model '{model_name}'. If your timm build uses "
            "different CLIP names, try one of: "
            "'vit_base_patch16_clip_224.openai', "
            "'vit_base_patch16_clip_224.laion2b', or inspect with "
            "timm.list_models('*clip*vit_base*patch16*')."
        )
        raise RuntimeError(msg) from exc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Train CLIP ViT-B/16 on FF++ + all extra datasets")
    print("=" * 88)
    print(f"Device      : {device}")
    print(f"AMP enabled : {amp_enabled}")
    print(f"Model       : {args.model_name}")
    print(f"Image size  : {args.image_size}")
    print(f"LR          : {args.lr}")
    print(f"Manifest    : {args.manifest}")
    print(f"Root        : {args.root_dir}")
    print(f"Save dir    : {save_dir}")

    train_df, val_df = prepare_video_split(args)
    train_df = cap_df(train_df, args.max_train_frames, args.seed)
    val_df = cap_df(val_df, args.max_val_frames, args.seed + 1)
    print("\nAfter optional frame caps:")
    print(f"  train frames: {len(train_df)}  real={(train_df['label'] == 0).sum()}  fake={(train_df['label'] == 1).sum()}")
    print(f"  val frames  : {len(val_df)}  real={(val_df['label'] == 0).sum()}  fake={(val_df['label'] == 1).sum()}")

    train_set = FrameDataset(train_df, args.image_size, train=True)
    val_set = FrameDataset(val_df, args.image_size, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_clip_vit(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=make_class_weight(train_df, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_auc = -math.inf
    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 88)
        print(f"Epoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.3e}")
        print("=" * 88)
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, amp_enabled, epoch)
        print(f"  mean train loss: {loss:.4f}")

        val_auc, val_ap, val_acc = evaluate(model, val_loader, device, amp_enabled, "Combined Val")
        scheduler.step()

        state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auc": val_auc,
            "val_ap": val_ap,
            "val_acc": val_acc,
            "args": vars(args),
        }
        torch.save(state, save_dir / "last.pth")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(state, save_dir / "best.pth")
            print(f"  saved new best: {save_dir / 'best.pth'}  val_auc={best_auc:.4f}")

    print("\nDone.")
    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
