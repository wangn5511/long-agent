# 对batch1和bach2中的 wave_in 和 wave_out类别的clip的frame进行前向移动,以扩展数据的多样性, 
# 最大的前移帧数为min(4, floor(0.4*clip_len))  因为模型输入的clip长度为8 
# "NONE": 0,  # 没有手势
# "WAVE_IN": 1,  # 内挥
# "WAVE_OUT": 2,  # 外挥

file_path = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_3_第一批V2手势-更新-rewrite_balanced_train_filtered_clip_None.txt"
output_file_path = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_3_第一批V2手势-更新-rewrite_balanced_train_filtered_clip_None_move_forward.txt"

def move_forward_frames(input_file, output_file):
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
        
        # 只处理label为1和2的行
        if label not in [1, 2]:
            processed_lines.append(line)
            continue
        
        # 计算最大前移帧数
        max_shift = min(4, int(0.4 * length))
        
        # 生成前移后的行
        for shift in range(max_shift + 1):
            new_start_frame = start_frame - shift
            if new_start_frame < 0:
                continue  # 避免负的起始帧
            processed_lines.append(f"{video_path} {new_start_frame} {length} {label}")
    
    # 写入处理后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')

move_forward_frames(file_path, output_file_path)

