import os
import random
import shutil
from pathlib import Path

def split_dataset(
    src_root: str,
    dst_root: str,
    train_ratio: float = 0.9,
    seed: int = 42
):
    random.seed(seed)
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 创建目标根目录
    dst_train = dst_root / "train"
    dst_val = dst_root / "val"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)

    # 遍历每个类别文件夹（0,1,2,...）
    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        print(f"处理类别: {class_name}")

        files = [f for f in class_dir.iterdir() if f.is_file()]
        if not files:
            print(f"  类别 {class_name} 没有文件，跳过")
            continue

        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:]

        # 创建目标类别子目录
        train_class_dir = dst_train / class_name
        val_class_dir = dst_val / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # 拷贝文件
        for f in train_files:
            shutil.copy2(f, train_class_dir / f.name)
        for f in val_files:
            shutil.copy2(f, val_class_dir / f.name)

        print(f"  总数: {n_total}, train: {len(train_files)}, val: {len(val_files)}")

if __name__ == "__main__":
    # 源数据目录
    src_dir = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/recordings_new"
    # 目标目录（会自动创建）
    dst_dir = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_process/recordings_new_split"

    split_dataset(src_dir, dst_dir, train_ratio=0.9, seed=42)
    print("完成！")