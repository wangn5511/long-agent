import sys
import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append("/home/fanchen/ultralytics-main")

def train_model():
    """训练模型的函数"""
    # 加载模型
    model = YOLO("/home/fanchen/ultralytics-main/yolov12n.pt")  # 加载预训练模型
    
    # 开始训练
    results = model.train(
        data='/home/fanchen/ultralytics-main/ultralytics/cfg/datasets/open_images_hand_head.yaml',
        epochs=300,
        imgsz=640,
        batch=1024,
        name='my_detection_model',
        device=[0, 1, 2, 3],
        workers=64,
        patience=300,
        project='my_project_head_hand',
        save=True,
        cache=True,
    )
    
    # 验证模型
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    
    return model

def inference_single_image(model_path, image_path, save_output=True, show_result=False):
    """单张图像推理函数
    
    Args:
        model_path: 训练好的模型路径
        image_path: 要推理的图像路径
        save_output: 是否保存输出图像
        show_result: 是否显示结果
    
    Returns:
        results: YOLO推理结果对象
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 执行推理
    print(f"正在处理图像: {image_path}")
    results = model(image_path)
    
    # 获取检测结果
    for r in results:
        boxes = r.boxes  # 边界框
        masks = r.masks  # 分割掩码（如果适用）
        probs = r.probs  # 分类概率（如果适用）
        
        # 打印检测到的对象信息
        print(f"检测到 {len(boxes)} 个对象")
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id] if hasattr(model, 'names') else f'Class {class_id}'
            print(f"对象 {i+1}: {class_name}, 置信度: {confidence:.2f}")
    
    # 可视化结果
    plotted = results[0].plot()  # 获取绘制了检测框的图像
    
    # 保存结果
    if save_output:
        os.makedirs("/home/fanchen/ultralytics-main/results/", exist_ok=True)
        output_path = "/home/fanchen/ultralytics-main/results/" + image_path.split('/')[-1].replace('.', '_result.')
        cv2.imwrite(output_path, plotted)
        print(f"结果已保存到: {output_path}")
    
    # 显示结果
    if show_result:
        plt.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('目标检测结果')
        plt.show()
    
    return results

if __name__ == '__main__':
    # 取消注释下面的行来训练模型
    # model = train_model()
    
    # 单张图像推理
    # 请替换为您自己的模型路径和图像路径
    model_path = '/home/fanchen/ultralytics-main/runs/pose/yolo12_hand_keypoints16/weights/best.pt'  # 训练好的最佳模型
    image_path = '/home/datasets/gesture_dataset/gesture_v1_new/images/WAVE_IN/recording_20250902_152815/frame_0047.png'  # 替换为您的测试图像路径
    
    # 执行推理
    inference_single_image(model_path, image_path, save_output=True, show_result=False)