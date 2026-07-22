import os
import shutil
from pathlib import Path
from tqdm import tqdm

# python /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/process_labeled.py

# 定义源目录和目标目录
SOURCE_DIR = "/home/models/hagrid_processed"
TARGET_DIR = "/home/models/hagrid_labeled_add_10"

# 定义标签映射关系
LABEL_MAPPING = {
    # "four": "4",
    # "like": "7",
    # "no_gesture": "0",
    # "one": "1",
    # "ok": "6",
    # "palm": "5",
    # "peace": "2",
    # "peace_inverted": "2",
    # "three": "3",
    # "three2": "3"
    "thumb_index": "8",
    "fist": "9",
}

# 数据集类型
DATASET_TYPES = ["train", "test", "val"]


def create_target_directories():
    """创建目标目录结构"""
    for dataset_type in DATASET_TYPES:
        for label_num in set(LABEL_MAPPING.values()):
            target_path = Path(TARGET_DIR) / dataset_type / label_num
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {target_path}")


def copy_images():
    """复制图片到对应的目标目录"""
    total_copied = 0
    
    for dataset_type in DATASET_TYPES:
        print(f"\n处理 {dataset_type} 数据集...")
        
        for gesture_label, numeric_label in LABEL_MAPPING.items():
            source_path = Path(SOURCE_DIR) / dataset_type / gesture_label
            target_path = Path(TARGET_DIR) / dataset_type / numeric_label
            
            # 检查源目录是否存在
            if not source_path.exists():
                print(f"警告: 源目录不存在 - {source_path}")
                continue
            
            # 获取所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            image_files = [f for f in source_path.iterdir() 
                          if f.is_file() and f.suffix in image_extensions]
            
            if not image_files:
                print(f"  {gesture_label} -> {numeric_label}: 没有找到图片")
                continue
            
            # 复制图片
            copied_count = 0
            for image_file in tqdm(image_files, 
                                  desc=f"  {gesture_label} -> {numeric_label}",
                                  leave=False):
                try:
                    target_file = target_path / image_file.name
                    # 如果目标文件已存在，添加前缀避免覆盖
                    if target_file.exists():
                        target_file = target_path / f"{gesture_label}_{image_file.name}"
                    
                    shutil.copy2(image_file, target_file)
                    copied_count += 1
                except Exception as e:
                    print(f"\n错误: 复制文件 {image_file} 失败 - {e}")
            
            print(f"  {gesture_label} -> {numeric_label}: 复制了 {copied_count} 张图片")
            total_copied += copied_count
    
    return total_copied


def main():
    """主函数"""
    print("=" * 60)
    print("开始处理图片标签重组任务")
    print("=" * 60)
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"标签映射: {LABEL_MAPPING}")
    print("=" * 60)
    
    # 创建目标目录结构
    print("\n步骤 1: 创建目标目录结构")
    create_target_directories()
    
    # 复制图片
    print("\n步骤 2: 复制图片到目标目录")
    total_copied = copy_images()
    
    # 完成提示
    print("\n" + "=" * 60)
    print(f"任务完成! 总共复制了 {total_copied} 张图片")
    print("=" * 60)


if __name__ == "__main__":
    main()