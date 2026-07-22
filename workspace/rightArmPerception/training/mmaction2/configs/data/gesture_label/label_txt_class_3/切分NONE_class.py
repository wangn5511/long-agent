#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def process_label_file(input_file, output_file):
    """
    处理标签文件，对label为0的行进行数据切分
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) != 4:
            # 如果不是标准格式，直接保留原行
            processed_lines.append(line)
            continue
            
        video_path, start_frame, length, label = parts
        start_frame = int(start_frame)
        length = int(length)
        label = int(label)
        
        # 只处理label为0的行
        if label != 0:
            processed_lines.append(line)
            continue
        
        # 对label为0的行进行切分
        # 新的length固定为10
        new_length = 16
        step = 20  # start_frame递增步长
        
        # 生成切分后的行
        current_start = start_frame
        while current_start + new_length <= start_frame + length:
            # 确保不会超出原始范围
            processed_lines.append(f"{video_path} {current_start} {new_length} {label}")
            current_start += step
            
            # 如果下一步会超出范围，则停止
            if current_start + new_length > start_frame + length:
                break
    
    # 写入处理后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')


if __name__ == "__main__":
    input_file = "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_test_filtered.txt"
    output_file = "/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite_balanced_test_filtered_clip_None.txt"
    
    process_label_file(input_file, output_file)
    print(f"处理完成，结果已保存到: {output_file}")