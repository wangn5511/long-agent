import os
import random
from pathlib import Path

"""
将指定目录中merged、val和test文件夹下的0类别数据只保留十分之一，随机抽取，其余的删除。
"""

def reduce_class0_data(
    root_dir: str,
    class_name: str = "0",
    keep_ratio: float = 0.1,
    seed: int = 42,
):
    """
    对merged、val和test文件夹中的指定类别数据随机保留一定比例，删除其余文件。
    
    Args:
        root_dir: 根目录路径，包含merged、val、test三个文件夹
        class_name: 要处理的类别名称，默认为"0"
        keep_ratio: 保留的比例，默认为0.1（十分之一）
        seed: 随机种子，默认为42
    """
    random.seed(seed)
    root = Path(root_dir)
    
    # 要处理的文件夹列表
    folders = ["merged", "val", "test"]
    
    for folder_name in folders:
        class_dir = root / folder_name / class_name
        
        if not class_dir.exists():
            print(f"警告: {class_dir} 不存在，跳过")
            continue
        
        if not class_dir.is_dir():
            print(f"警告: {class_dir} 不是目录，跳过")
            continue
        
        # 获取所有文件
        files = [f for f in class_dir.iterdir() if f.is_file()]
        
        if not files:
            print(f"{folder_name}/{class_name}: 没有文件，跳过")
            continue
        
        total_count = len(files)
        keep_count = max(1, int(total_count * keep_ratio))  # 至少保留1个文件
        
        # 随机打乱并选择要保留的文件
        random.shuffle(files)
        keep_files = set(files[:keep_count])
        delete_files = files[keep_count:]
        
        # 删除不需要的文件
        deleted_count = 0
        for file_path in delete_files:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  删除文件失败 {file_path}: {e}")
        
        print(f"{folder_name}/{class_name}: 总数={total_count}, 保留={keep_count}, 删除={deleted_count}")
    
    print("处理完成！")


if __name__ == "__main__":
    root_directory = "/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random"
    reduce_class0_data(root_directory, class_name="0", keep_ratio=0.1, seed=42)

