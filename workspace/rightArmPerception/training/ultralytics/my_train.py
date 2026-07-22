import sys
import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
print("脚本绝对路径:", script_path)

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
print("脚本所在目录:", script_dir)

sys.path.append(script_dir)


from ultralytics import YOLO



import argparse

parser = argparse.ArgumentParser(description="Train a YOLO model.")
parser.add_argument('--dataset', type=str, default=f'{script_dir}/ultralytics/cfg/datasets/open_images_hand_head.yaml', help='Path to dataset configuration file (e.g., data.yaml)')
parser.add_argument('--model', type=str, default=f'{script_dir}/ultralytics/cfg/models/12/yolo12n-head_hand.yaml', help='Path to model configuration file (e.g., model.yaml)')
parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained weights file (e.g., yolov8n.pt)')
parser.add_argument('--device', type=list, default=[0, 1, 2, 3], help='Device to use for training (e.g., 0,1,2,3 or cpu)')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
parser.add_argument('--batch', type=int, default=1024, help='Batch size for training')

args = parser.parse_args()



if args.weights is not None:
# Load a model
    model = YOLO(args.model).load(args.weights)  # build a new model from YAML
else:
    model = YOLO(args.model)  # build a new model from YAML


# 开始训练
results = model.train(
    data=args.dataset,  # 数据集配置文件路径
    epochs=args.epochs,                        # 训练轮数
    imgsz=args.imgsz,                         # 图像大小
    batch=args.batch,                          # 批次大小
    name='my_detection_model',         # 实验名称，用于保存结果
    device=args.device,                        # 使用的设备，0表示第一个GPU，'cpu'表示使用CPU
    workers=8,                         # 数据加载工作线程数
    patience=300,                       # 早停耐心值
    project='my_project_head_hand',              # 项目名称，用于组织实验
    save=True,                         # 保存模型
    cache=True,                        # 缓存数据到内存以加速训练
)

# 验证模型
metrics = model.val()
print(metrics.box.map)  # 打印 mAP50-95

# 导出模型（可选）
# model.export(format='onnx')  # 导出为 ONNX 格式