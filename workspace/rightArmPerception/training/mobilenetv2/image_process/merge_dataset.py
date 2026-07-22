import os
import shutil
from pathlib import Path
from tqdm import tqdm

# python /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/merge_dataset.py

# 定义基础目录
BASE_DIR = "/home/models/hagrid_labeled_add_10"

# 定义要合并的数据集类型
SOURCE_DATASETS = ["train", "val"]

# 定义标签文件夹（数字标签）
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 目标合并文件夹
MERGED_DIR = "merged"


def create_merged_directories():
    """创建merged目录结构"""
    print("\n创建merged目录结构...")
    merged_path = Path(BASE_DIR) / MERGED_DIR
    
    for label in LABELS:
        label_path = merged_path / label
        label_path.mkdir(parents=True, exist_ok=True)
        print(f"  创建目录: {label_path}")


def merge_datasets():
    """合并train和val数据集到merged文件夹"""
    total_copied = 0
    stats = {}
    
    for label in LABELS:
        print(f"\n处理标签 {label}...")
        target_path = Path(BASE_DIR) / MERGED_DIR / label
        copied_count = 0
        
        for dataset_type in SOURCE_DATASETS:
            source_path = Path(BASE_DIR) / dataset_type / label
            
            # 检查源目录是否存在
            if not source_path.exists():
                print(f"  警告: 源目录不存在 - {source_path}")
                continue
            
            # 获取所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            image_files = [f for f in source_path.iterdir() 
                          if f.is_file() and f.suffix in image_extensions]
            
            if not image_files:
                print(f"  {dataset_type}/{label}: 没有找到图片")
                continue
            
            # 复制图片
            for image_file in tqdm(image_files, 
                                  desc=f"  从 {dataset_type}/{label} 复制",
                                  leave=False):
                try:
                    target_file = target_path / image_file.name
                    
                    # 如果目标文件已存在，添加数据集类型前缀避免覆盖
                    if target_file.exists():
                        target_file = target_path / f"{dataset_type}_{image_file.name}"
                    
                    shutil.copy2(image_file, target_file)
                    copied_count += 1
                except Exception as e:
                    print(f"\n  错误: 复制文件 {image_file} 失败 - {e}")
            
            print(f"  从 {dataset_type}/{label} 复制了 {len(image_files)} 张图片")
        
        stats[label] = copied_count
        total_copied += copied_count
        print(f"  标签 {label} 总计: {copied_count} 张图片")
    
    return total_copied, stats


def print_statistics(total_copied, stats):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("合并统计信息:")
    print("=" * 60)
    
    for label in LABELS:
        count = stats.get(label, 0)
        print(f"  标签 {label}: {count:>6} 张图片")
    
    print("-" * 60)
    print(f"  总计:    {total_copied:>6} 张图片")
    print("=" * 60)


def verify_merge():
    """验证合并后的数据"""
    print("\n验证merged文件夹...")
    merged_path = Path(BASE_DIR) / MERGED_DIR
    
    for label in LABELS:
        label_path = merged_path / label
        if label_path.exists():
            image_count = len([f for f in label_path.iterdir() 
                             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            print(f"  标签 {label}: {image_count} 张图片")
        else:
            print(f"  标签 {label}: 目录不存在")


def main():
    """主函数"""
    print("=" * 60)
    print("开始合并train和val数据集")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"源数据集: {', '.join(SOURCE_DATASETS)}")
    print(f"目标文件夹: {MERGED_DIR}")
    print(f"标签: {', '.join(LABELS)}")
    print("=" * 60)
    
    # 步骤1: 创建merged目录结构
    print("\n步骤 1: 创建目录结构")
    create_merged_directories()
    
    # 步骤2: 合并数据集
    print("\n步骤 2: 合并数据集")
    total_copied, stats = merge_datasets()
    
    # 步骤3: 打印统计信息
    print_statistics(total_copied, stats)
    
    # 步骤4: 验证合并结果
    verify_merge()
    
    print("\n" + "=" * 60)
    print("任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

