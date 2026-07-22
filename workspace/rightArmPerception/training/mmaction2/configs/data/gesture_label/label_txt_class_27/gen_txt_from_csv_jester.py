import os
import csv

# 路径设置
labels_csv = '/home/datasets/21bn-jester/jester-v1-labels.csv'
train_csv = '/home/datasets/21bn-jester/jester-v1-train.csv'
raw_video_dir = '/home/datasets/21bn-jester/raw_data'
output_txt = '/home/fanchen/hmimodels-main/label_txt_class_27/jester_train_list.txt'

# 1. 加载类别名称到 ID 的映射
label_to_id = {}
with open(labels_csv, 'r') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        label_name = row[0]
        label_to_id[label_name] = idx

print(f"Loaded {len(label_to_id)} labels.")
print(label_to_id)

# 2. 处理训练集 CSV 并统计帧数
results = []

with open(train_csv, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        video_name = row[0].strip()
        class_name = row[1].strip()

        # 获取类别 ID
        if class_name not in label_to_id:
            print(f"[Warning] Unknown label '{class_name}' for video {video_name}")
            continue
        class_id = label_to_id[class_name]

        # 计算帧数（即文件夹下的图片数量）
        video_path = os.path.join(raw_video_dir, video_name)
        if not os.path.isdir(video_path):
            print(f"[Warning] Video folder does not exist: {video_path}")
            continue

        # 获取所有图片文件
        frame_files = [
            name for name in os.listdir(video_path)
            if name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        frame_count = len(frame_files)

        if frame_count == 0:
            print(f"[Warning] No frames found in video folder: {video_path}")
            continue

        # 检查帧命名是否符合5位数格式（从00001到00001+frame_count-1）
        expected_frames = set([f"{i:05d}.jpg" for i in range(1, frame_count + 1)])
        actual_frames = set([f.lower() for f in frame_files])
        
        # 检查是否有缺失的帧
        missing_frames = expected_frames - actual_frames
        extra_frames = actual_frames - expected_frames
        
        if missing_frames:
            print(f"[Warning] Missing frames in {video_path}: {sorted(missing_frames)[:10]}{'...' if len(missing_frames) > 10 else ''}")
            continue  # 如果缺帧，跳过该视频
        
        if extra_frames:
            print(f"[Warning] Extra frames in {video_path}: {sorted(extra_frames)[:10]}{'...' if len(extra_frames) > 10 else ''}")
            continue
        
        # 如果帧命名不符合预期格式，可以选择跳过或者给出警告
        if missing_frames or extra_frames:
            print(f"[Warning] Frame naming issue in {video_path}. Expected {frame_count} frames numbered 00001.jpg to {frame_count:05d}.jpg")
            continue

        # 添加记录
        results.append(f"{video_name} 1 {frame_count - 1} {class_id}")

# 3. 写入输出文件
with open(output_txt, 'w') as f:
    f.write('\n'.join(results))

print(f"Successfully written {len(results)} entries to {output_txt}")