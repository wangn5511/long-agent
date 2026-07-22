import os
import json
import argparse
from PIL import Image


def load_frame_bboxes(json_path: str):
    """
    从 mmpose 导出的 json 中读取每一帧的 bbox 信息

    返回:
        frame_to_bboxes: dict[int, list[tuple[float,float,float,float]]]
            例如 {1: [(x1,y1,x2,y2), (x1,y1,x2,y2), ...], 2: [...], ...}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_to_bboxes = {}

    instance_info = data.get("instance_info", [])
    for frame_item in instance_info:
        frame_id = int(frame_item.get("frame_id"))
        bboxes = []
        for inst in frame_item.get("instances", []):
            # json 中 bbox 是一个 list，里面又包了一层，例如 [[x1,y1,x2,y2]]
            bbox_list = inst.get("bbox", [])
            if not bbox_list:
                continue
            # 只取第一项
            x1, y1, x2, y2 = bbox_list[0]
            bboxes.append((x1, y1, x2, y2))

        if bboxes:
            frame_to_bboxes[frame_id] = bboxes

    return frame_to_bboxes


def crop_and_save_images(frame_to_bboxes,
                         img_dir: str,
                         out_dir: str,
                         img_prefix: str = "frame_",
                         img_ext: str = ".png",
                         expand_pixels: int = 50):
    """
    根据 frame_to_bboxes 中的框，对 img_dir 下的图片进行裁剪并保存到 out_dir。

    裁剪方式：
        先按原始 bbox 计算中心点 (cx, cy)，
        然后在上下左右各扩展 expand_pixels 像素，
        最后裁剪到图像边界内。
    """
    os.makedirs(out_dir, exist_ok=True)

    for frame_id, bboxes in sorted(frame_to_bboxes.items()):
        img_name = f"{img_prefix}{frame_id:04d}{img_ext}"
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] 图像不存在，跳过: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
            # 原始 bbox 的中心点和宽高的一半
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = (x2 - x1) / 2.0 + expand_pixels
            half_h = (y2 - y1) / 2.0 + expand_pixels

            # 扩展后的 bbox
            left_f = cx - half_w
            right_f = cx + half_w
            top_f = cy - half_h
            bottom_f = cy + half_h

            # 转成整数并裁剪到图像范围内
            left = max(0, int(round(left_f)))
            top = max(0, int(round(top_f)))
            right = min(w, int(round(right_f)))
            bottom = min(h, int(round(bottom_f)))

            if right <= left or bottom <= top:
                print(
                    f"[WARN] 非法扩展 bbox, frame {frame_id}, inst {idx}: "
                    f"orig=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                    f"expanded=({left},{top},{right},{bottom})，已跳过"
                )
                continue

            cropped = img.crop((left, top, right, bottom))

            out_name = f"{img_prefix}{frame_id:04d}_inst_{idx:02d}{img_ext}"
            out_path = os.path.join(out_dir, out_name)
            cropped.save(out_path)

        print(f"[INFO] 处理完成 frame_id={frame_id}, 共裁剪 {len(bboxes)} 个实例")


def main():
    parser = argparse.ArgumentParser(
        description="根据 mmpose json 中的 bbox 裁剪每帧图片（并在中心上下左右扩展固定像素）"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=False,
        default="/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/mmpose_result_recording_20250917_123720.json",
        help="mmpose 结果 json 文件路径"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/recording_20250917_123720",
        help="原始帧图像所在目录（frame_0001.png 等）"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/processd",
        help="裁剪后图片输出目录"
    )
    parser.add_argument(
        "--expand_pixels",
        type=int,
        default=50,
        help="从中心点向上下左右扩展的像素数（默认 50）"
    )
    args = parser.parse_args()

    frame_to_bboxes = load_frame_bboxes(args.json_path)
    print(f"[INFO] 共解析到 {len(frame_to_bboxes)} 帧的 bbox 信息")

    crop_and_save_images(
        frame_to_bboxes=frame_to_bboxes,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        img_prefix="frame_",
        img_ext=".png",
        expand_pixels=args.expand_pixels,
    )


if __name__ == "__main__":
    main()
