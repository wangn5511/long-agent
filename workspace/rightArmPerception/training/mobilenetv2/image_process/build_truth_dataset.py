#!/usr/bin/env python3
"""
依据最新真实录制数据与 HaGRID 原始标注数据，构建融合后的数据集：
1. 将 recordings_new_split 中 merged/val 的 0-7 类全部复制到目标路径。
2. 统计每个 split/class 的样本数量。
3. 按 1:100 的比例，从 hagrid_labeled 中抽取对应数量的数据补充到目标路径。
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CLASSES = [str(i) for i in range(8)]
SPLITS = ["merged", "val"]


@dataclass
class CopyStats:
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def add(self, split: str, cls: str, amount: int) -> None:
        self.counts[(split, cls)] = self.counts.get((split, cls), 0) + amount

    def get(self, split: str, cls: str) -> int:
        return self.counts.get((split, cls), 0)

    def report(self) -> str:
        lines: List[str] = []
        for split in SPLITS:
            lines.append(f"[{split}]")
            for cls in CLASSES:
                lines.append(f"  class {cls}: {self.get(split, cls)} samples")
        return "\n".join(lines)


def list_files(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    return [p for p in folder.iterdir() if p.is_file()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_with_unique_name(src: Path, dst_dir: Path, prefix: str = "") -> None:
    ensure_dir(dst_dir)
    base_name = f"{prefix}{src.name}"
    target = dst_dir / base_name
    if not target.exists():
        shutil.copy2(src, target)
        return

    stem = target.stem
    suffix = target.suffix
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return
        idx += 1


def copy_split_classes(
    src_root: Path, dst_root: Path, stats: CopyStats, prefix: str = ""
) -> None:
    for split in SPLITS:
        for cls in CLASSES:
            src_cls = src_root / split / cls
            dst_cls = dst_root / split / cls
            files = list_files(src_cls)
            for f in files:
                copy_with_unique_name(f, dst_cls, prefix=prefix)
            stats.add(split, cls, len(files))


def sample_and_copy(
    src_root: Path,
    dst_root: Path,
    stats: CopyStats,
    ratio: int,
    rng: random.Random,
    prefix: str,
) -> None:
    for split in SPLITS:
        for cls in CLASSES:
            required = stats.get(split, cls) * ratio
            if required == 0:
                continue
            src_cls = src_root / split / cls
            files = list_files(src_cls)
            if not files:
                continue

            if required >= len(files):
                chosen = files
                actual = len(files)
                print(
                    f"警告: {split}/{cls} 仅有 {len(files)} 个 HaGRID 样本，"
                    f"低于期望 {required}，将全部复制。"
                )
            else:
                chosen = rng.sample(files, required)
                actual = required
            for f in chosen:
                copy_with_unique_name(f, dst_root / split / cls, prefix=prefix)
            stats.add(split, cls, actual)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 1:100 规则构建 hagrid_truth_merged 数据集"
    )
    parser.add_argument(
        "--new-data",
        type=Path,
        default=Path("/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_new_split"),
        help="新录制数据根目录（含 merged/test）",
    )
    parser.add_argument(
        "--labeled-data",
        type=Path,
        default=Path("/home/models/hagrid_labeled"),
        help="HaGRID 原始标注数据根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/models/hagrid_truth_merged_20"),
        help="输出目录，将复制 merged/test 结构",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=20,
        help="新增数据与补充数据的比例（默认为 1:20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证抽样可复现",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    stats = CopyStats()

    print(">>> 复制新录制数据到目标目录 ...")
    copy_split_classes(args.new_data, args.output, stats, prefix="")

    print(">>> 按 1:{} 抽样 HaGRID 标注数据 ...".format(args.ratio))
    sample_and_copy(
        args.labeled_data,
        args.output,
        stats,
        ratio=args.ratio,
        rng=rng,
        prefix="hagrid_",
    )

    print(">>> 完成，最终样本统计：")
    print(stats.report())


if __name__ == "__main__":
    main()

