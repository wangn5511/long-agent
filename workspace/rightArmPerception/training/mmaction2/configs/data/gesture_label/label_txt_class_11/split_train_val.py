import os
import random
from collections import defaultdict

# 设置随机种子以确保结果可重现
random.seed(42)

def split_train_test_balanced(input_file, train_output_file, test_output_file, test_ratio=0.1):
    """
    将输入文件按类别分组，然后确保测试集中每个类别的样本数量相当
    
    参数:
        input_file: 输入文件路径
        train_output_file: 训练集输出文件路径
        test_output_file: 测试集输出文件路径
        test_ratio: 测试集比例，默认为0.1
    """
    # 按类别分组数据
    class_data = defaultdict(list)
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 按类别分组
    for line in lines:
        line = line.strip()
        if line:
            # 提取类别（最后一个数字）
            parts = line.split(" ")
            if len(parts) >= 4:
                class_id = parts[-1]
                class_data[class_id].append(line)
    
    # 统计每个类别的数量
    class_counts = {cls: len(data) for cls, data in class_data.items()}
    print("每个类别的样本数量:")
    for cls, count in sorted(class_counts.items()):
        print(f"类别 {cls}: {count} 个样本")
    
    # 计算测试集中应该包含的总样本数
    total_samples = len(lines)
    total_test_samples = int(total_samples * test_ratio)
    
    # 计算每个类别在测试集中应该包含的样本数
    num_classes = len(class_data)
    # 确保每个类别的测试样本数不超过其总样本数
    max_test_samples_per_class = min(total_test_samples // num_classes, min(class_counts.values()))
    
    # 对于可能剩余的测试样本数，平均分配给部分类别
    remaining_test_samples = total_test_samples - (max_test_samples_per_class * num_classes)
    
    # 打乱每个类别的数据
    for cls in class_data:
        random.shuffle(class_data[cls])
    
    train_data = []
    test_data = []
    
    # 为每个类别分配测试样本
    for i, (cls, data) in enumerate(sorted(class_data.items())):
        # 基础测试样本数
        test_samples_needed = max_test_samples_per_class
        # 分配剩余的测试样本
        if i < remaining_test_samples:
            test_samples_needed += 1
        
        # 确保不超过该类别的总样本数
        test_samples_needed = min(test_samples_needed, len(data))
        
        # 分配到测试集和训练集
        test_data.extend(data[:test_samples_needed])
        train_data.extend(data[test_samples_needed:])
    
    # 再次打乱训练集和测试集的顺序
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 保存训练集
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    # 保存测试集
    with open(test_output_file, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + '\n')
    
    # 统计训练集和测试集中每个类别的数量
    train_class_counts = defaultdict(int)
    for line in train_data:
        parts = line.split()
        if len(parts) >= 4:
            train_class_counts[parts[-1]] += 1
    
    test_class_counts = defaultdict(int)
    for line in test_data:
        parts = line.split()
        if len(parts) >= 4:
            test_class_counts[parts[-1]] += 1
    
    # 输出统计信息
    print(f"\n总共样本数: {total_samples}")
    print(f"训练集样本数: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"测试集样本数: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")
    print(f"目标测试集比例: {test_ratio*100:.1f}%")
    
    print(f"\n每个类别在测试集中的样本数目标: {max_test_samples_per_class} 或 {max_test_samples_per_class+1}")
    
    print("\n训练集中每个类别的数量:")
    for cls in sorted(class_data.keys()):
        print(f"类别 {cls}: {train_class_counts.get(cls, 0)}")
    
    print("\n测试集中每个类别的数量:")
    for cls in sorted(class_data.keys()):
        print(f"类别 {cls}: {test_class_counts.get(cls, 0)}")
    
    return len(train_data), len(test_data)

if __name__ == "__main__":
    # 定义文件路径
    input_file = '/home/fanchen/hmimodels-main/label_txt_class_11/第一批V1手势-更新_rewrite.txt'
    train_output_file = '/home/fanchen/hmimodels-main/label_txt_class_11/第一批V1手势-更新_rewrite_balanced_train.txt'
    test_output_file = '/home/fanchen/hmimodels-main/label_txt_class_11/第一批V1手势-更新_rewrite_balanced_test.txt'
    
    # 执行划分
    print("开始划分训练集和测试集(平衡模式)...")
    train_count, test_count = split_train_test_balanced(input_file, train_output_file, test_output_file)
    print(f"\n划分完成！")
    print(f"训练集已保存到: {train_output_file}")
    print(f"测试集已保存到: {test_output_file}")