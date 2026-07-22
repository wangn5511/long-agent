
label_mapping = {
    "NONE": 0,  # 没有手势
    "WAVE_IN": 1,  # 内挥
    "WAVE_OUT": 2,  # 外挥
    # "OK": 3,  # OK 手势
    # "WAVE_UP": 4,  # 上挥
    # "WAVE_DOWN": 5,  # 下挥
    # "WAVE_LEFT": 6,  # 左挥
    # "WAVE_RIGHT": 7,  # 右挥
    # "BYEBYE": 8,  # "byebye" 挥
    # "THUMBS_UP": 9,  # 点赞    THUMBS_UP: 自建数据集名称     THUNBS UP: batch1和batch2名称
    # "FIST_MOVE": 10,  # 握拳 + 前后移动
}

class_mapping = {idx: class_name for class_name, idx in label_mapping.items()}

txt_file = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_11_第一批V2手势-更新-rewrite_balanced_test_filtered_clip_None.txt"
output_txt_file = "/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_3_第一批V2手势-更新-rewrite_balanced_test_filtered_clip_None.txt"


with open(txt_file, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
filtered_lines = []


if "batch" not in txt_file:
    for line in lines:
        line = line.strip().split(" ")   # file_name offset length label
        if line[0].split("/")[0] in label_mapping.keys():  # label 保持一致
            line[-1] = str(label_mapping[line[0].split("/")[0]])
            filtered_lines.append(" ".join(line))
        else:  # 如果label不在映射中，默认归为0（NONE）
            line[-1] = "0"
            filtered_lines.append(" ".join(line))
else:
    for line in lines:
        line = line.strip().split(" ")   # file_name offset length label
        if int(line[-1]) in class_mapping.keys():  # label 保持一致
            filtered_lines.append(" ".join(line))
        else:  # 如果label不在映射中，默认归为0（NONE）
            line[-1] = "0"
            filtered_lines.append(" ".join(line))
        
        
with open(output_txt_file, mode='w', encoding='utf-8') as out_txtfile:
    for line in filtered_lines:
        out_txtfile.write(line + "\n")
print(f"过滤并重写标签文件完成，结果已保存到: {output_txt_file}")
