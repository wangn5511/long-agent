#!/usr/bin/env python3
"""Build RKNN calibration assets for the 4-input gated RGB+skeleton fusion model.

Strategy: drive the **actual val pipeline** of the training config, dump the
post-pipeline tensors as four ``.npy`` files per sample, and write a manifest
where each line lists the four input file paths in the **exact ONNX input
order**:

    keypoint  imgs  handedness  global_traj

This guarantees the calibration data matches what the model sees during
training/inference (same DecodeVideoROI, same Hand21WorldNormalize, same
global-traj formatting, same handedness embedding lookup), which is the
single most important property for INT8 quantisation scale selection.

Notes on each output:
  * keypoint    : (1, 16, 21, 4) float32 - wrist-relative, scale-normalised,
                  4th channel is the per-joint score.
  * imgs        : (1, 16, 3, 96, 96) float32, **raw [0, 255]** - the ONNX
                  graph applies the ImageNet mean/std internally, so we must
                  NOT pre-normalise here.
  * handedness  : (1,) int64 - 0/1/2 for Left/Right/Unknown.
  * global_traj : (1, 16, 5) float32 - delta xyz + absolute xy.

The class order is read from the config's ``categories`` list, so this script
works for both the 9-class and 10-class variants without modification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        required=True,
        help="Path to the training config used to train the model "
             "(the val_pipeline drives the dump).",
    )
    p.add_argument(
        "--repo-root",
        default="/home/yangzhao/code/rightArmPerception/training/mmaction2",
        help="Inserted to sys.path so handoff.* custom modules resolve.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for npy files, manifest, and summary.json.",
    )
    p.add_argument(
        "--num-per-class",
        type=int,
        default=8,
        help="Number of samples to dump per class. Default 8.",
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="If > 0, cap the total number of samples written (after "
             "per-class balancing).",
    )
    p.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="Which split of the dataset to dump from. Default val "
             "(deterministic sampling).",
    )
    p.add_argument(
        "--manifest-name",
        default="stgcn_3d_rgb_gated_calibration_dataset.txt",
    )
    p.add_argument(
        "--summary-name",
        default="summary.json",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for per-class sample selection.",
    )
    return p.parse_args()


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def main() -> None:
    args = parse_args()

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # Import mmengine + mmaction lazily so import errors surface clearly.
    from mmengine.config import Config
    from mmengine.registry import DATASETS, init_default_scope
    from mmaction.utils import register_all_modules

    register_all_modules()
    init_default_scope("mmaction")

    cfg = Config.fromfile(args.config)
    categories: List[str] = list(cfg.categories)
    num_classes = len(categories)
    print(f"[info] config: {args.config}")
    print(f"[info] num_classes={num_classes}  categories={categories}")

    # Pull the underlying PoseDataset (un-wrap any RepeatDataset).
    if args.split == "val":
        ds_cfg = cfg.val_dataloader.dataset
    else:
        ds_cfg = cfg.train_dataloader.dataset
    # Strip RepeatDataset wrapper if present.
    while ds_cfg.get("type") == "RepeatDataset":
        ds_cfg = ds_cfg.dataset
    ds = DATASETS.build(ds_cfg)
    print(f"[info] dataset built: {type(ds).__name__}  len={len(ds)}")

    # Group sample indices by label (use ann_info, do NOT iterate full pipeline).
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        label = int(info.get("label", -1))
        if 0 <= label < num_classes:
            by_class[label].append(i)
    for c in range(num_classes):
        print(f"  {c:>2d} {categories[c]:<12s}: {len(by_class.get(c, []))} samples available")

    # Balanced selection.
    rng = np.random.default_rng(args.seed)
    selected: List[Tuple[int, int]] = []  # (sample_idx, label)
    for c in range(num_classes):
        pool = by_class.get(c, [])
        if not pool:
            print(f"[warn] class {c} ({categories[c]}) has 0 samples - skipped.")
            continue
        n = min(args.num_per_class, len(pool))
        picks = rng.choice(len(pool), size=n, replace=False)
        for p in picks:
            selected.append((pool[int(p)], c))

    if args.max_total > 0 and len(selected) > args.max_total:
        keep_idx = rng.choice(len(selected), size=args.max_total, replace=False)
        selected = [selected[int(i)] for i in keep_idx]

    print(f"[info] selected {len(selected)} samples to dump")

    out_dir = Path(args.out_dir).expanduser().resolve()
    sub_dirs = {n: out_dir / n for n in ("keypoint", "imgs", "handedness", "global_traj")}
    for d in sub_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest_lines: List[str] = []
    summary_records: List[dict] = []

    skipped = 0
    for k, (idx, label) in enumerate(selected):
        try:
            item = ds[idx]
        except Exception as e:
            print(f"[warn] sample {idx} failed: {e!r}")
            skipped += 1
            continue

        inputs = item["inputs"]
        kp = inputs["keypoint"]
        imgs = inputs["imgs"]
        # tensors → np
        if hasattr(kp, "numpy"):
            kp = kp.numpy()
        if hasattr(imgs, "numpy"):
            imgs = imgs.numpy()

        kp = np.asarray(kp, dtype=np.float32)        # (1, T, V, C)
        imgs = np.asarray(imgs, dtype=np.float32)    # (T, 3, H, W)
        # Add batch dim to imgs so it matches ONNX input layout (N, T, 3, H, W).
        if imgs.ndim == 4:
            imgs = imgs[None, ...]                   # (1, T, 3, H, W)

        ds_sample = item["data_samples"]
        traj = np.asarray(ds_sample.global_traj, dtype=np.float32)  # (1, T, C_traj)
        if traj.ndim == 2:
            traj = traj[None, ...]
        meta = ds_sample.metainfo
        hand_id = int(meta.get("handedness_label_id", 2))
        hand = np.array([hand_id], dtype=np.int64)

        video_name = sanitize(str(meta.get("video_name", f"idx{idx}")))
        stem = f"{categories[label]}__{video_name}__{idx:06d}"

        paths = {
            "keypoint": sub_dirs["keypoint"] / f"{stem}.npy",
            "imgs": sub_dirs["imgs"] / f"{stem}.npy",
            "handedness": sub_dirs["handedness"] / f"{stem}.npy",
            "global_traj": sub_dirs["global_traj"] / f"{stem}.npy",
        }
        np.save(paths["keypoint"], kp)
        np.save(paths["imgs"], imgs)
        np.save(paths["handedness"], hand)
        np.save(paths["global_traj"], traj)

        # IMPORTANT: order must match ONNX input order.
        manifest_lines.append(
            f"{paths['keypoint']} {paths['imgs']} {paths['handedness']} {paths['global_traj']}"
        )
        summary_records.append({
            "dataset_idx": idx,
            "label": label,
            "label_name": categories[label],
            "video_name": meta.get("video_name"),
            "track_id": meta.get("track_id"),
            "handedness_label_id": hand_id,
            "shapes": {
                "keypoint": list(kp.shape),
                "imgs": list(imgs.shape),
                "handedness": list(hand.shape),
                "global_traj": list(traj.shape),
            },
            "value_ranges": {
                "keypoint_min": float(kp.min()), "keypoint_max": float(kp.max()),
                "imgs_min": float(imgs.min()), "imgs_max": float(imgs.max()),
                "global_traj_min": float(traj.min()), "global_traj_max": float(traj.max()),
            },
            "paths": {k: str(v) for k, v in paths.items()},
        })

        if (k + 1) % 10 == 0:
            print(f"  dumped {k + 1}/{len(selected)}")

    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    per_class_counts = defaultdict(int)
    for r in summary_records:
        per_class_counts[r["label_name"]] += 1

    summary = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "num_classes": num_classes,
        "categories": categories,
        "num_selected": len(selected),
        "num_written": len(summary_records),
        "num_skipped": skipped,
        "num_per_class_requested": args.num_per_class,
        "per_class_written": dict(per_class_counts),
        "manifest_path": str(manifest_path.resolve()),
        "samples": summary_records,
    }
    (out_dir / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary : {out_dir / args.summary_name}")
    print(f"[done] dumped  : {len(summary_records)} samples (skipped: {skipped})")


if __name__ == "__main__":
    main()
