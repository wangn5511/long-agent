#!/usr/bin/env python3
"""
将 GUN 目录下的数据按照 9:1 的比例复制到目标目录
- 90% 的数据复制到 merged/9
- 10% 的数据复制到 test/9
"""

import os
import random
import shutil
from pathlib import Path


def copy_gun_data(
    src_dir: str,
    dst_merged: str,
    dst_test: str,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """
    将源目录下的所有文件按照指定比例复制到两个目标目录
    
    Args:
        src_dir: 源目录路径（GUN目录）
        dst_merged: 目标merged目录（90%的数据）
        dst_test: 目标test目录（10%的数据）
        train_ratio: 训练集比例，默认为0.9（90%）
        seed: 随机种子，默认为42
    """
    random.seed(seed)
    
    src_path = Path(src_dir)
    merged_path = Path(dst_merged)
    test_path = Path(dst_test)
    
    # 检查源目录是否存在
    if not src_path.exists():
        raise ValueError(f"源目录不存在: {src_dir}")
    
    if not src_path.is_dir():
        raise ValueError(f"源路径不是目录: {src_dir}")
    
    # 创建目标目录
    merged_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有文件（只包括文件，不包括目录）
    files = [f for f in src_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"警告: {src_dir} 中没有找到文件")
        return
    
    # 随机打乱文件列表
    random.shuffle(files)
    
    # 计算分割点
    n_total = len(files)
    n_merged = int(n_total * train_ratio)
    merged_files = files[:n_merged]
    test_files = files[n_merged:]
    
    # 复制文件到merged目录
    print(f"正在复制 {len(merged_files)} 个文件到 merged/9...")
    for file_path in merged_files:
        try:
            shutil.copy2(file_path, merged_path / file_path.name)
        except Exception as e:
            print(f"  复制文件失败 {file_path.name}: {e}")
    
    # 复制文件到test目录
    print(f"正在复制 {len(test_files)} 个文件到 test/9...")
    for file_path in test_files:
        try:
            shutil.copy2(file_path, test_path / file_path.name)
        except Exception as e:
            print(f"  复制文件失败 {file_path.name}: {e}")
    
    print(f"\n处理完成！")
    print(f"总文件数: {n_total}")
    print(f"merged/9: {len(merged_files)} 个文件")
    print(f"test/9: {len(test_files)} 个文件")
    print(f"比例: {len(merged_files)/n_total*100:.1f}% : {len(test_files)/n_total*100:.1f}%")


if __name__ == "__main__":
    # 源目录
    source_directory = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/FIST"
    
    # 目标目录
    destination_merged = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2/merged/8"
    destination_test = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2/test/8"
    
    copy_gun_data(
        src_dir=source_directory,
        dst_merged=destination_merged,
        dst_test=destination_test,
        train_ratio=0.9,
        seed=42
    )

