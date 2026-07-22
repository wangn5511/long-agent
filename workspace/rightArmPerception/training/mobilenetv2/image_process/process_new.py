import os
import json
import argparse
from PIL import Image
from typing import List, Dict, Tuple


def load_hagrid_annotations(json_path: str, target_label: str):
    """
    从 HaGRID 数据集的 annotation json 中读取指定 label 的 bbox 信息
    
    Args:
        json_path: annotation json 文件路径
        target_label: 要提取的目标 label（如 "no_gesture", "four" 等）
    
    Returns:
        image_data: dict[str, list[tuple[float,float,float,float]]]
            键为图片ID（不含扩展名），值为该图片中所有目标label的bbox列表
            bbox 格式为归一化坐标 (x_center, y_center, width, height)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_data = {}
    
    for img_id, info in data.items():
        bboxes = info.get("bboxes", [])
        labels = info.get("labels", [])
        
        # 筛选出目标 label 的 bbox
        target_bboxes = []
        for bbox, label in zip(bboxes, labels):
            if label == target_label:
                target_bboxes.append(tuple(bbox))
        
        if target_bboxes:
            image_data[img_id] = target_bboxes
    
    return image_data


def normalize_bbox_to_pixels(bbox: Tuple[float, float, float, float],
                             img_width: int,
                             img_height: int,
                             expand_ratio: float = 0.1) -> Tuple[int, int, int, int]:
    """
    将归一化的 bbox 坐标转换为像素坐标，并进行扩展
    
    Args:
        bbox: (x_left, y_top, width, height) 归一化坐标 (0-1)
              x_left: 左上角X坐标，y_top: 左上角Y坐标
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）
        expand_ratio: 扩展比例，相对于 bbox 的宽高（默认 0.1 = 10%）
    
    Returns:
        (left, top, right, bottom) 像素坐标
    """
    x_left, y_top, width, height = bbox
    
    # 转换为像素坐标
    x_left_px = x_left * img_width
    y_top_px = y_top * img_height
    w_px = width * img_width
    h_px = height * img_height
    
    # 计算中心点
    cx_px = x_left_px + w_px / 2.0
    cy_px = y_top_px + h_px / 2.0
    
    # 扩展
    half_w = w_px / 2.0 * (1 + expand_ratio)
    half_h = h_px / 2.0 * (1 + expand_ratio)
    
    # 计算边界
    left = max(0, int(round(cx_px - half_w)))
    top = max(0, int(round(cy_px - half_h)))
    right = min(img_width, int(round(cx_px + half_w)))
    bottom = min(img_height, int(round(cy_px + half_h)))
    
    return left, top, right, bottom


def process_hagrid_dataset(labels_to_process: List[str],
                          img_base_dir: str,
                          annotation_base_dir: str,
                          output_base_dir: str,
                          expand_ratio: float = 0.1):
    """
    处理 HaGRID 数据集，裁剪指定 label 的图片
    
    Args:
        labels_to_process: 要处理的 label 列表，如 ["four", "like"]
        img_base_dir: 图片根目录 /home/datasets/gesture_datasets/hagrid
        annotation_base_dir: annotation 根目录 /home/datasets/gesture_datasets/annotations
        output_base_dir: 输出根目录 /home/pub_envs/hagrid_processed
        expand_ratio: bbox 扩展比例（默认 0.1）
    """
    splits = ["train", "val", "test"]
    
    total_processed = 0
    total_cropped = 0
    
    for label in labels_to_process:
        print(f"\n{'='*60}")
        print(f"[INFO] 开始处理 label: {label}")
        print(f"{'='*60}")
        
        # 图片所在目录
        img_dir = os.path.join(img_base_dir, label)
        
        if not os.path.exists(img_dir):
            print(f"[WARN] 图片目录不存在，跳过: {img_dir}")
            continue
        
        for split in splits:
            print(f"\n[INFO] 处理 {label} - {split} 集")
            
            # annotation json 路径
            json_path = os.path.join(annotation_base_dir, split, f"{label}.json")
            
            if not os.path.exists(json_path):
                print(f"[WARN] annotation 文件不存在，跳过: {json_path}")
                continue
            
            # 输出目录
            output_dir = os.path.join(output_base_dir, split, label)
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载 annotations
            image_data = load_hagrid_annotations(json_path, label)
            print(f"[INFO] 从 {split} 集中找到 {len(image_data)} 张包含 '{label}' 的图片")
            
            # 处理每张图片
            processed_count = 0
            cropped_count = 0
            
            for img_id, bboxes in image_data.items():
                img_path = os.path.join(img_dir, f"{img_id}.jpg")
                
                if not os.path.exists(img_path):
                    # print(f"[WARN] 图片不存在，跳过: {img_path}")
                    continue
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size
                    
                    # 对每个 bbox 进行裁剪
                    for idx, bbox in enumerate(bboxes, start=1):
                        left, top, right, bottom = normalize_bbox_to_pixels(
                            bbox, w, h, expand_ratio
                        )
                        
                        if right <= left or bottom <= top:
                            print(
                                f"[WARN] 非法 bbox: {img_id}, bbox {idx}: "
                                f"norm={bbox}, pixels=({left},{top},{right},{bottom})"
                            )
                            continue
                        
                        cropped = img.crop((left, top, right, bottom))
                        
                        # 保存裁剪的图片
                        out_name = f"{img_id}_bbox_{idx:02d}.jpg"
                        out_path = os.path.join(output_dir, out_name)
                        cropped.save(out_path, quality=95)
                        
                        cropped_count += 1
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"  已处理 {processed_count}/{len(image_data)} 张图片...")
                
                except Exception as e:
                    print(f"[ERROR] 处理图片 {img_path} 时出错: {e}")
                    continue
            
            print(f"[INFO] {split} 集处理完成: 处理了 {processed_count} 张图片，裁剪了 {cropped_count} 个区域")
            total_processed += processed_count
            total_cropped += cropped_count
    
    print(f"\n{'='*60}")
    print(f"[INFO] 全部处理完成！")
    print(f"[INFO] 总共处理了 {total_processed} 张图片，裁剪了 {total_cropped} 个区域")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="处理 HaGRID 数据集，根据 annotations 裁剪指定 label 的图片"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["four", "like"],
        help="要处理的 label 列表，如: --labels four like no_gesture"
    )
    parser.add_argument(
        "--img_base_dir",
        type=str,
        default="/home/datasets/gesture_datasets/hagrid",
        help="图片根目录"
    )
    parser.add_argument(
        "--annotation_base_dir",
        type=str,
        default="/home/datasets/gesture_datasets/annotations",
        help="annotation 根目录"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/home/pub_envs/hagrid_processed",
        help="输出根目录"
    )
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=0.1,
        help="bbox 扩展比例，相对于原始宽高（默认 0.1 = 10%）"
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] 要处理的 labels: {args.labels}")
    print(f"[INFO] 图片目录: {args.img_base_dir}")
    print(f"[INFO] Annotation 目录: {args.annotation_base_dir}")
    print(f"[INFO] 输出目录: {args.output_base_dir}")
    print(f"[INFO] 扩展比例: {args.expand_ratio}")
    
    process_hagrid_dataset(
        labels_to_process=args.labels,
        img_base_dir=args.img_base_dir,
        annotation_base_dir=args.annotation_base_dir,
        output_base_dir=args.output_base_dir,
        expand_ratio=args.expand_ratio
    )


if __name__ == "__main__":
    main()
