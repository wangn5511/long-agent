import argparse
from pathlib import Path


def compare_frame_dirs(src_dir: str, dst_dir: str):
    """
    对比两个根目录下的所有抽帧图片，检查个数和文件名是否一致。
    目录结构: <root>/<label>/<video>/frame_XXXX.png

    Args:
        src_dir: 源目录
        dst_dir: 目标目录
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        print(f"[✗] Source directory does not exist: {src_dir}")
        return
    if not dst_dir.exists():
        print(f"[✗] Destination directory does not exist: {dst_dir}")
        return

    # 收集两边的相对路径（相对于各自根目录）
    src_files = {p.relative_to(src_dir) for p in src_dir.rglob("*.png")}
    dst_files = {p.relative_to(dst_dir) for p in dst_dir.rglob("*.png")}

    # ---- 1. 总数对比 ----
    print("=" * 60)
    print("Overall comparison")
    print("=" * 60)
    print(f"  Source ({src_dir}): {len(src_files)} files")
    print(f"  Dest   ({dst_dir}): {len(dst_files)} files")

    if src_files == dst_files:
        print("\n[✓] Both directories have identical file sets.")
        return

    # ---- 2. 差异分析 ----
    only_in_src = sorted(src_files - dst_files)
    only_in_dst = sorted(dst_files - src_files)

    # 按 label/video 维度汇总
    src_videos = {p.parent for p in src_files}   # label/video
    dst_videos = {p.parent for p in dst_files}

    videos_only_in_src = sorted(src_videos - dst_videos)
    videos_only_in_dst = sorted(dst_videos - src_videos)
    common_videos = sorted(src_videos & dst_videos)

    print("\n" + "=" * 60)
    print("Video-level (label/video) diff")
    print("=" * 60)

    if videos_only_in_src:
        print(f"\n[!] {len(videos_only_in_src)} video(s) only in SOURCE:")
        for v in videos_only_in_src:
            count = sum(1 for f in src_files if f.parent == v)
            print(f"    {v}/  ({count} frames)")

    if videos_only_in_dst:
        print(f"\n[!] {len(videos_only_in_dst)} video(s) only in DEST:")
        for v in videos_only_in_dst:
            count = sum(1 for f in dst_files if f.parent == v)
            print(f"    {v}/  ({count} frames)")

    if not videos_only_in_src and not videos_only_in_dst:
        print("  All video directories match.")

    # ---- 3. 共有 video 内的帧差异 ----
    frame_diff_videos = []
    for v in common_videos:
        src_frames = sorted(f.name for f in src_files if f.parent == v)
        dst_frames = sorted(f.name for f in dst_files if f.parent == v)
        if src_frames != dst_frames:
            frame_diff_videos.append((v, src_frames, dst_frames))

    if frame_diff_videos:
        print("\n" + "=" * 60)
        print("Frame-level diff in common videos")
        print("=" * 60)
        for v, sf, df in frame_diff_videos:
            src_set = set(sf)
            dst_set = set(df)
            only_src = sorted(src_set - dst_set)
            only_dst = sorted(dst_set - src_set)
            print(f"\n  {v}/  src={len(sf)} frames, dst={len(df)} frames")
            if only_src:
                preview = only_src[:5]
                print(f"    Only in src ({len(only_src)}): {preview}{'...' if len(only_src) > 5 else ''}")
            if only_dst:
                preview = only_dst[:5]
                print(f"    Only in dst ({len(only_dst)}): {preview}{'...' if len(only_dst) > 5 else ''}")
    else:
        if common_videos:
            print("\n[✓] All common videos have identical frame sets.")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Files only in src: {len(only_in_src)}")
    print(f"  Files only in dst: {len(only_in_dst)}")
    if not only_in_src and not only_in_dst:
        print("  [✓] Directories are identical.")
    else:
        print("  [✗] Directories differ.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two frame directories for consistency."
    )
    parser.add_argument("--src_dir", type=str, required=True,
                        help="源目录，结构为 <src>/<label>/<video>/frame_XXXX.png")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="目标目录，结构为 <dst>/<label>/<video>/frame_XXXX.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_frame_dirs(args.src_dir, args.dst_dir)
