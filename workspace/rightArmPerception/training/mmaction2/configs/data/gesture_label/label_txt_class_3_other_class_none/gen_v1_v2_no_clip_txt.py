
label_mapping = {
    "NONE": 0,  # 没有手势
    "WAVE_IN": 1,  # 内挥
    "WAVE_OUT": 2,  # 外挥
    "OK": 3,  # OK 手势
    "WAVE_UP": 4,  # 上挥
    "WAVE_DOWN": 5,  # 下挥
    "WAVE_LEFT": 6,  # 左挥
    "WAVE_RIGHT": 7,  # 右挥
    "BYEBYE": 8,  # "byebye" 挥
    "THUMBS_UP": 9,  # 点赞    THUMBS_UP: 自建数据集名称     THUNBS UP: batch1和batch2名称
    "FIST_MOVE": 10,  # 握拳 + 前后移动
    # "NUMBER": 11,  # 数字手势 1–5
    # "SWIPE_FORWARD": 12,  # 前滑动（桌面视角）
    # "SWIPE_BACKWARD": 13,  # 后滑动（桌面视角）
    # "SWIPE_LEFT": 14,  # 左滑动（桌面视角）
    # "SWIPE_RIGHT": 15  # 右滑动（桌面视角）
}

class_mapping = {idx: class_name for class_name, idx in label_mapping.items()}

# 读取txt的没一行：file_name offset length label; 统计不同file_name的个数
txt_path = f"/home/fanchen/hmimodels-main/label_txt_class_11/gesture_v2_rewrite.txt"
file_name_set = {}
with open(txt_path, mode='r', encoding='utf-8') as txtfile:
    lines = txtfile.readlines()
    for line in lines:
        line = line.strip().split(" ")   # file_name offset length label
        if line[0].split("/")[0] in label_mapping.keys():  # label要变一下
            line[-1] = str(label_mapping[line[0].split("/")[0]])
            file_name_set[line[0]] = line[1:]


print(f"不同的file_name个数为：{len(file_name_set)}")


txt_path_new = f"/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_11_第一批V2手势-更新.txt"
new_file_name_set = {}
with open(txt_path_new, mode='r', encoding='utf-8') as txtfile:
    lines = txtfile.readlines()
    for line in lines:
        line = line.strip().split(" ")   # file_name offset length label
        new_file_name_set[line[0]] = line[1:]

# 若new_file中的key在file中，则替换file中的value
for key in new_file_name_set.keys():
    if key in file_name_set.keys():
        file_name_set[key] = new_file_name_set[key]
    else:
        file_name_set[key] = new_file_name_set[key]
        print(f"key {key} not in file_name_set")



out_txt_path = f"/home/fanchen/hmimodels-main/label_txt_class_3_other_class_none/class_11_第一批V2手势-更新_rewrite.txt"
with open(out_txt_path, mode='w', encoding='utf-8') as out_txtfile:
    for key in file_name_set.keys():
        line = f"{key} {' '.join(file_name_set[key])}\n"
        out_txtfile.write(line)






