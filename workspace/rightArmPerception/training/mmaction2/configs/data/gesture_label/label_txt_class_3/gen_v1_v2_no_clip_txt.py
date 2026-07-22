
class_mapping = {
    0: "NONE",
    1: "WAVE_IN",
    2: "WAVE_OUT",
}

label_mapping = {class_name: idx for idx, class_name in class_mapping.items()}
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


txt_path_new = f"/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels.txt"
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



out_txt_path = f"/home/fanchen/hmimodels-main/label_txt_class_3/第一批V2手势-更新-video_labels_rewrite.txt"
with open(out_txt_path, mode='w', encoding='utf-8') as out_txtfile:
    for key in file_name_set.keys():
        line = f"{key} {' '.join(file_name_set[key])}\n"
        out_txtfile.write(line)






