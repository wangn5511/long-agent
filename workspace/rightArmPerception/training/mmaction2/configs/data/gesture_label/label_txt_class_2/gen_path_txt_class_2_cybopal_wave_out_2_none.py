import csv
import os
import random

#  固定seed，确保可复现
random.seed(42)

input_csv = "/home/datasets/rest_gesture_dataset/annotations_filled.csv"
output_txt = "/home/fanchen/hmimodels-main/label_txt_class_2/cybopal_camera_process-video_labels_wave_out_2_none.txt"

train_txt = "/home/fanchen/hmimodels-main/label_txt_class_2/cybopal_camera_process-video_labels_train_wave_out_2_none.txt"
test_txt = "/home/fanchen/hmimodels-main/label_txt_class_2/cybopal_camera_process-video_labels_test_wave_out_2_none.txt"

# 训练集和测试集的比例，
train_ratio = 0.9
test_ratio = 1 - train_ratio

# 要求测试集各类别数量基本一样
os.makedirs(os.path.dirname(output_txt), exist_ok=True)


# label_mapping = {
#     "NONE": 0,  # 没有手势
#     "OK": 1,  # OK 手势
#     "WAVE_IN": 2,  # 内挥
#     "WAVE_OUT": 3,  # 外挥
#     "WAVE_UP": 4,  # 上挥
#     "WAVE_DOWN": 5,  # 下挥
#     "WAVE_LEFT": 6,  # 左挥
#     "WAVE_RIGHT": 7,  # 右挥
#     "BYEBYE": 8,  # "byebye" 挥
#     "THUNBS UP": 9,  # 点赞
#     "FIST_MOVE": 10,  # 握拳 + 前后移动
#     "NUMBER": 11,  # 数字手势 1–5
#     "SWIPE_FORWARD": 12,  # 前滑动（桌面视角）
#     "SWIPE_BACKWARD": 13,  # 后滑动（桌面视角）
#     "SWIPE_LEFT": 14,  # 左滑动（桌面视角）
#     "SWIPE_RIGHT": 15  # 右滑动（桌面视角）
# }




# label_mapping = {
#     "NONE": 0,  # 没有手势
#     "OK": 1,  # OK 手势
#     "WAVE_IN": 2,  # 内挥
#     "WAVE_OUT": 3,  # 外挥
#     "WAVE_UP": 4,  # 上挥
#     "WAVE_DOWN": 5,  # 下挥
#     "WAVE_LEFT": 6,  # 左挥
#     "WAVE_RIGHT": 7,  # 右挥
#     "BYEBYE": 8,  # "byebye" 挥
#     "THUMBS_UP": 9,  # 点赞
#     "FIST_MOVE": 10,  # 握拳 + 前后移动
#     # "NUMBER": 11,  # 数字手势 1–5
#     # "SWIPE_FORWARD": 12,  # 前滑动（桌面视角）
#     # "SWIPE_BACKWARD": 13,  # 后滑动（桌面视角）
#     # "SWIPE_LEFT": 14,  # 左滑动（桌面视角）
#     # "SWIPE_RIGHT": 15  # 右滑动（桌面视角）
# }



class_mapping = {
    0: "NONE",
    1: "WAVE_IN",
    2: "WAVE_OUT",
}

label_mapping = {class_name: idx for idx, class_name in class_mapping.items()}

# images_root = f"/home/datasets/gesture_dataset/batch1-更新/images/"
images_root = ""
all_data = []

cnt = 0

with open(input_csv, mode='r', encoding='utf-8') as csvfile:

    reader = csv.reader(csvfile)
    next(reader)  # 如果有表头就跳过第一行

    for row in reader:
        video_name = row[0].split('.')[0]
        start_offset = max(int(float(row[1])), 1)
        length = int(float(row[2])) - start_offset
        label = row[-1]

        if label not in label_mapping:
            print(f"未找到标签 {label}，跳过 {video_name}")
            cnt += 1
            continue
        if label == "WAVE_OUT":
            all_data.append((video_name, start_offset, length, label_mapping["NONE"]))
        else:
            all_data.append((video_name, start_offset, length, label_mapping[label]))



# # 额外处理NONE这类目标
# if 'batch' not in input_csv:
#     images_root = f"/home/datasets/gesture_dataset/fc_v2/images/NONE"
#     for i_file_name in os.listdir(images_root):
#         if os.path.isdir(f"{images_root}/{i_file_name}"):
#             # 去除其中以‘.’开头的图像文件
#             tem_png = sorted([i for i in os.listdir(f"{images_root}/{i_file_name}") if not i.startswith(".")])
#             length = len(tem_png) - 1
#             all_data.append((f"NONE/{i_file_name}", 1, length, label_mapping["NONE"]))



with open(output_txt, mode='w', encoding='utf-8') as txtfile:
    for data in all_data:
        txtfile.write(f"{data[0]} {data[1]} {data[2]} {data[3]}\n")

# 随机选择0.9的训练集合
random.shuffle(all_data)

train_data = all_data[:int(train_ratio * len(all_data))]
test_data = all_data[int(train_ratio * len(all_data)):]

# 确保测试集各类别数量基本一样
test_class_count = {}
for data in test_data:
    class_name = class_mapping[data[3]]
    test_class_count[class_name] = test_class_count.get(class_name, 0) + 1

print("测试集各类别数量:", test_class_count)
# 写入训练集和测试集
with open(train_txt, mode='w', encoding='utf-8') as txtfile:
    for data in train_data:
        txtfile.write(f"{data[0]} {data[1]} {data[2]} {data[3]}\n")

with open(test_txt, mode='w', encoding='utf-8') as txtfile:
    for data in test_data:
        txtfile.write(f"{data[0]} {data[1]} {data[2]} {data[3]}\n")



# , \
#     #  open(output_txt, mode='w', encoding='utf-8') as txtfile

print("✅ 标签文件已成功生成！")
print(f"跳过 {cnt} 条数据")