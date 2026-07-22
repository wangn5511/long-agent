import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge_batches(
    batch1_dir: str,
    batch2_dir: str,
    batch1_xlsx: str,
    batch2_xlsx: str,
    output_dir: str,
    output_xlsx: str,
):
    """
    合并两个 batch 目录和对应的标注 xlsx 文件。

    目录结构:
        batch1-更新/  001/ 002/ ... 010/
        batch2/       011/ 012/ ... 020/
    合并后:
        output_dir/   001/ 002/ ... 020/
    """
    batch1_dir = Path(batch1_dir)
    batch2_dir = Path(batch2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Step 1: 合并目录 ==========
    print("=" * 60)
    print("Step 1: Merging directories")
    print("=" * 60)

    copied = 0
    skipped = 0
    for src_batch, batch_name in [(batch1_dir, "batch1"), (batch2_dir, "batch2")]:
        if not src_batch.exists():
            print(f"[✗] Directory not found: {src_batch}")
            continue

        subdirs = sorted([d for d in src_batch.iterdir() if d.is_dir()])
        print(f"\n[{batch_name}] {src_batch} -> {len(subdirs)} subdirectories")

        for subdir in subdirs:
            dst_subdir = output_dir / subdir.name
            if dst_subdir.exists():
                print(f"  [!] Skipping (already exists): {subdir.name}/")
                skipped += 1
                continue

            shutil.copytree(subdir, dst_subdir)
            file_count = sum(1 for _ in dst_subdir.rglob("*") if _.is_file())
            print(f"  [✓] Copied: {subdir.name}/  ({file_count} files)")
            copied += 1

    print(f"\nDirectory merge done: {copied} copied, {skipped} skipped")

    # 列出合并后的结果
    merged_subdirs = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    print(f"Merged directory contains {len(merged_subdirs)} subdirectories: {merged_subdirs}")

    # ========== Step 2: 合并 xlsx ==========
    print("\n" + "=" * 60)
    print("Step 2: Merging xlsx annotation files")
    print("=" * 60)

    df1 = pd.read_excel(batch1_xlsx)
    df2 = pd.read_excel(batch2_xlsx)

    print(f"  {batch1_xlsx}")
    print(f"    shape: {df1.shape}, columns: {list(df1.columns)}")
    print(f"  {batch2_xlsx}")
    print(f"    shape: {df2.shape}, columns: {list(df2.columns)}")

    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_excel(output_xlsx, index=False)

    print(f"\n  Merged shape: {merged_df.shape}")
    print(f"  [✓] Saved to {output_xlsx}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two batch directories and their annotation xlsx files."
    )
    parser.add_argument("--batch1_dir", type=str, required=True,
                        help="第一个 batch 目录路径 (e.g. batch1-更新)")
    parser.add_argument("--batch2_dir", type=str, required=True,
                        help="第二个 batch 目录路径 (e.g. batch2)")
    parser.add_argument("--batch1_xlsx", type=str, required=True,
                        help="第一个 batch 的标注 xlsx 文件")
    parser.add_argument("--batch2_xlsx", type=str, required=True,
                        help="第二个 batch 的标注 xlsx 文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="合并后的输出目录")
    parser.add_argument("--output_xlsx", type=str, required=True,
                        help="合并后的标注 xlsx 输出路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_batches(
        batch1_dir=args.batch1_dir,
        batch2_dir=args.batch2_dir,
        batch1_xlsx=args.batch1_xlsx,
        batch2_xlsx=args.batch2_xlsx,
        output_dir=args.output_dir,
        output_xlsx=args.output_xlsx,
    )
