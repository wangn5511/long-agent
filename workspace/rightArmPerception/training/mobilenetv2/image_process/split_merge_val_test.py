import os
import random
import shutil
from pathlib import Path
from typing import Tuple

"""
将 src_root 下的按类别组织的数据集按 7:2:1 划分为 merge/val/test 三个集合。
目录结构假设为：
src_root/
  class_a/
    img1.jpg
    ...
  class_b/
    ...
输出到 dst_root/
  merge/class_a/*
  val/class_a/*
  test/class_a/*
"""

def split_merge_val_test(
    src_root: str,
    dst_root: str,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    copy: bool = True,
):
    random.seed(seed)
    src = Path(src_root)
    dst = Path(dst_root)

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 必须相加为 1.0，当前为 {ratios} 总和 {sum(ratios)}")

    # 创建目标根目录
    dst_merge = dst / "merged"
    dst_val = dst / "val"
    dst_test = dst / "test"
    for d in (dst_merge, dst_val, dst_test):
        d.mkdir(parents=True, exist_ok=True)

    # 遍历每个类别文件夹
    class_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"处理类别: {class_name}")

        files = [f for f in class_dir.iterdir() if f.is_file()]
        if not files:
            print(f"  类别 {class_name} 没有文件，跳过")
            continue

        random.shuffle(files)
        n_total = len(files)
        n_merge = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        # 余数全部给 test，确保总数不丢失
        n_test = n_total - n_merge - n_val

        merge_files = files[:n_merge]
        val_files = files[n_merge:n_merge + n_val]
        test_files = files[n_merge + n_val:]

        # 创建目标类别子目录
        merge_class_dir = dst_merge / class_name
        val_class_dir = dst_val / class_name
        test_class_dir = dst_test / class_name
        for d in (merge_class_dir, val_class_dir, test_class_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 拷贝或移动文件
        op = shutil.copy2 if copy else shutil.move
        for f in merge_files:
            op(f, merge_class_dir / f.name)
        for f in val_files:
            op(f, val_class_dir / f.name)
        for f in test_files:
            op(f, test_class_dir / f.name)

        print(f"  总数: {n_total}, merge: {len(merge_files)}, val: {len(val_files)}, test: {len(test_files)}")

    print("划分完成！输出目录:", dst)


if __name__ == "__main__":
    src_dir = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all"
    dst_dir = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split"
    split_merge_val_test(src_dir, dst_dir, ratios=(0.7, 0.2, 0.1), seed=42, copy=True)
