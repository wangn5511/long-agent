import argparse
import os
import subprocess
from pathlib import Path


def prepare_gesture_image_data(
    src_dir: str,
    dst_dir: str,
    skip_convert: bool = False,
):
    """
    Prepare gesture data from source directory to destination directory.
    流程: 1) 先转换 MP4 格式 (libx264 + aac)  2) 从转换后的视频中抽帧

    Args:
        src_dir: 原始视频目录，结构为 <src_dir>/<label>/<video>.mp4
        dst_dir: 帧图片输出目录，输出结构为 <dst_dir>/<label>/<video>/frame_XXXX.png
                 - 如果 skip_convert=False: 还会创建 {dst_dir}_h264_aac/ 存放转码后的视频
        skip_convert: 如果为 True，跳过 MP4 格式转换，直接从 src_dir 抽帧到 dst_dir
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not skip_convert:
        # 中间目录：存放转换后的 h264/aac 视频
        converted_video_dir = dst_dir.parent / f"{dst_dir.name}_h264_aac"
        # 最终目录：存放抽取的帧（直接使用 dst_dir）
        frames_dir = dst_dir
        # ========== Step 1: 转换 MP4 格式 (libx264 + aac) ==========
        print("=" * 60)
        print("Step 1: Converting MP4 format (libx264 + aac)")
        print("=" * 60)

        for root, dirs, files in os.walk(src_dir):
            root_path = Path(root)
            rel_path = root_path.relative_to(src_dir)  # 相对于输入根目录的相对路径
            output_subdir = converted_video_dir / rel_path  # 保持相同的目录结构
            output_subdir.mkdir(parents=True, exist_ok=True)

            for file in files:
                if file.endswith(".mp4"):
                    input_file = root_path / file
                    output_file = output_subdir / file

                    # 如果输出文件已经存在，可以跳过
                    if output_file.exists():
                        print(f"[!] Skipping already converted: {output_file}")
                        continue

                    # 构建 ffmpeg 命令 - 转换格式
                    cmd = [
                        "ffmpeg",
                        "-i", str(input_file),
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-strict", "experimental",
                        "-y",  # 强制覆盖输出文件
                        str(output_file)
                    ]

                    print(f"[+] Converting: {input_file} -> {output_file}")
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if result.returncode != 0:
                        print(f"[✗] Failed: {input_file.name}\nError: {result.stderr.decode()}")
                    else:
                        print(f"[✓] Converted: {output_file}")

        # 抽帧时使用转换后的视频目录
        video_source_dir = converted_video_dir
    else:
        print("[!] Skipping MP4 format conversion (--skip_convert)")
        # 跳过转换时，直接从原始目录抽帧，输出到 dst_dir
        video_source_dir = src_dir
        frames_dir = dst_dir

    # ========== Step 2: 从视频中抽帧 ==========
    print("\n" + "=" * 60)
    print("Step 2: Extracting frames from videos")
    print("=" * 60)

    frames_dir.mkdir(parents=True, exist_ok=True)

    for video_path in video_source_dir.rglob("*.mp4"):
        label_name = video_path.parent.name  # 类别名
        video_name = video_path.stem  # 视频名（去掉 .mp4）

        # 输出目录为 /<frames_dir>/<label_name>/<video_name>/
        frame_output_dir = frames_dir / label_name / video_name

        # 如果输出目录已经存在且包含文件，可以跳过
        if frame_output_dir.exists() and any(frame_output_dir.iterdir()):
            print(f"[!] Skipping already extracted: {frame_output_dir}")
            continue

        frame_output_dir.mkdir(parents=True, exist_ok=True)

        # 构建 ffmpeg 命令 - 抽帧
        output_pattern = str(frame_output_dir / "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vsync", "0",
            output_pattern
        ]

        print(f"[+] Extracting: {video_path} -> {frame_output_dir}")

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[✓] Success: {video_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[✗] Failed: {video_path.name}")
            print(f"    Error: {e.stderr.decode('utf-8').strip()}")

    print("\n" + "=" * 60)
    print("Done!")
    if not skip_convert:
        print(f"  Converted videos: {converted_video_dir}")
    print(f"  Extracted frames: {frames_dir}")
    print("=" * 60)


def prepare_mmaction_rawframe_dataset_file():
    """TODO: Prepare dataset annotation file for mmaction rawframe format."""
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gesture video processing: convert MP4 format and extract frames."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- prepare_gesture_image_data ----
    sp = subparsers.add_parser(
        "extract_frames",
        help="Convert videos (optional) and extract frames. "
             "Input: <src_dir>/<label>/<video>.mp4  ->  Output: <dst>/<label>/<video>/frame_XXXX.png",
    )
    sp.add_argument("--src_dir", type=str, required=True,
                     help="原始视频目录，结构为 <src_dir>/<label>/<video>.mp4")
    sp.add_argument("--dst_dir", type=str, required=True,
                     help="帧图片输出目录，结构为 <dst_dir>/<label>/<video>/frame_XXXX.png")
    sp.add_argument("--no_skip_convert", dest="skip_convert", action="store_false",
                     help="不跳过 MP4 格式转换，先用 libx264+aac 转码再抽帧（默认跳过转换）")
    sp.set_defaults(skip_convert=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "extract_frames":
        prepare_gesture_image_data(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            skip_convert=args.skip_convert,
        )
    else:
        parse_args()  # will print help and exit
