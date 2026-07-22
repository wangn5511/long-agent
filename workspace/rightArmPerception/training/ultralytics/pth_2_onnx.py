import sys
sys.path.append("/home/fanchen/ultralytics-main")

import os
from ultralytics import YOLO

def convert_yolo_to_onnx(model_path, output_path, imgsz=224, batch=1):
    """
    将YOLO模型转换为ONNX格式
    
    参数:
        model_path: 输入的YOLO模型路径
        output_path: 输出的ONNX模型路径
        imgsz: 输入图像大小
        batch: 批处理大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"正在加载模型: {model_path}")
    # 加载YOLO模型
    model = YOLO(model_path)
    
    print(f"正在转换模型到ONNX格式，输入尺寸: {imgsz}...")
    # 使用YOLO内置的export方法转换为ONNX
    model.export(format="onnx", imgsz=imgsz, batch=batch, simplify=True)
    
    # YOLO默认会在原目录下生成ONNX文件，这里移动到指定位置
    default_onnx_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(default_onnx_path):
        os.rename(default_onnx_path, output_path)
        print(f"模型已成功转换并保存到: {output_path}")
    else:
        print(f"无法找到生成的ONNX文件: {default_onnx_path}")




if __name__ == "__main__":
    # 模型和输出路径
    model_path = "/home/fanchen/ultralytics-main/weights_yolov12n/best.pt"
    output_path = "/home/fanchen/ultralytics-main/weights_yolov12n/best_224.onnx"
    
    # 转换模型，指定输入尺寸为224
    convert_yolo_to_onnx(model_path, output_path, imgsz=224)

   