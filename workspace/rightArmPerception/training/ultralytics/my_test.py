import sys
sys.path.append("/home/fanchen/ultralytics-main")

from ultralytics import YOLO
import argparse
import cv2
from pathlib import Path
import time
import numpy as np

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12目标检测测试脚本')
    parser.add_argument('--model', type=str, 
                       default='/home/fanchen/ultralytics-main/runs/detect/my_project_head_hand/my_detection_model/weights/best.pt',
                       help='训练好的模型权重路径')
    parser.add_argument('--source', type=str,  
                       help='测试数据源，可以是图像路径、视频路径、图像目录或0(网络摄像头)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, 
                       help='NMS的IoU阈值')
    parser.add_argument('--save', action='store_true', 
                       help='是否保存检测结果')
    parser.add_argument('--show', action='store_true', 
                       help='是否显示检测结果')
    parser.add_argument('--save_txt', action='store_true', 
                       help='是否保存检测结果的文本文件')
    parser.add_argument('--device', type=str, default='0', 
                       help='运行设备，如0(第一个GPU)或cpu')
    return parser.parse_args()

def test_model(args):
    """测试YOLO模型"""
    # 加载模型
    print(f"正在加载模型: {args.model}")
    model = YOLO(args.model)
    print("模型加载完成")
    
    # 检查数据源是否有效
    source = args.source
    if source.isdigit():
        source = int(source)  # 网络摄像头
    
    print(f"开始检测，数据源: {source}")
    
    # 执行检测
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_txt=args.save_txt,
        show=args.show,
        device=args.device,
        project='runs/detect',
        name='test_results',
        stream=True if isinstance(source, int) else False  # 对于摄像头使用流模式
    )
    
    # 处理检测结果
    for result in results:
        # 获取检测信息
        boxes = result.boxes
        
        # 打印检测统计信息
        print(f"处理 {result.path if hasattr(result, 'path') else '帧'}: 检测到 {len(boxes)} 个目标")
        
        # 对于视频流或摄像头，按'q'键退出
        if args.show and isinstance(source, int):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("检测完成")
    if args.save:
        print(f"检测结果保存在: runs/detect/test_results")

def get_model_compute_stats(model, imgsz=640):
    """获取模型的计算量和参数量"""
    # 使用model.info()获取模型信息
    model_info = model.info(verbose=True)
    
    # 提取参数量和计算量信息
    # 如果model.info()直接返回字典
    if isinstance(model_info, dict):
        params = model_info.get('parameters', 0) / 1e6  # 转换为百万参数(M)
        flops = model_info.get('flops', 0) / 1e9  # 转换为十亿次浮点运算(G)
    else:
        # 对于较新版本的Ultralytics，model.info()可能直接打印信息
        # 我们手动计算参数量
        params = sum(p.numel() for p in model.parameters()) / 1e6  # 转换为百万参数(M)
        
        # 估算FLOPs（这是一个简化的估算）
        # 创建一个随机输入张量
        import torch
        input_shape = (1, 3, imgsz, imgsz)
        dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
        
        # 使用torchprofile或torchstat库计算FLOPs
        # 这里我们使用一个简化的方法，实际应用中可能需要更精确的计算
        try:
            from thop import profile
            flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
            flops = flops / 1e9  # 转换为十亿次浮点运算(G)
        except ImportError:
            print("警告: 未安装thop库，无法精确计算FLOPs")
            flops = None
    
    return params, flops

def measure_inference_time(model, imgsz=640, warmup_runs=5, test_runs=10):
    """测量模型的推理时间"""
    import torch
    
    # 创建随机输入
    device = next(model.parameters()).device
    input_shape = (1, 3, imgsz, imgsz)
    dummy_input = torch.randn(input_shape).to(device)
    
    # 预热运行
    print("正在进行预热...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.model(dummy_input)
    
    # 测试运行并计时
    print("正在测量推理时间...")
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            start_time = time.time()
            _ = model.model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def validate_model(model_path, data_path, device=[0, 1, 2, 3], imgsz=640):
    """验证模型性能，包括计算量和参数量统计"""
    print(f"正在验证模型性能: {model_path}")
    model = YOLO(model_path)
    
    # 获取模型参数量和计算量
    print("\n正在计算模型参数量和计算量...")
    params, flops = get_model_compute_stats(model, imgsz)
    print(f"模型参数量: {params:.2f} M")
    if flops is not None:
        print(f"模型计算量 (FLOPs): {flops:.2f} G")
    else:
        print("注意: 无法精确计算FLOPs，请安装thop库以获得更准确的结果")
    
    # 测量单张图像推理时间
    print("\n正在测量单张图像推理时间...")
    avg_time, std_time = measure_inference_time(model, imgsz)
    print(f"单张图像平均推理时间: {avg_time:.2f} ms ± {std_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")
    
    # 执行验证
    print("\n正在执行模型验证...")
    metrics = model.val(
        data=data_path,
        device=device,
        imgsz=imgsz
    )
    
    # 打印验证指标
    print("\n验证指标:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # 如果有特定类别的指标，可以打印
    if hasattr(metrics, 'names'):
        print("\n各类别指标:")
        for i, name in enumerate(metrics.names):
            print(f"{name}: mAP50-95 = {metrics.box.maps[i]:.4f}")
    
    return metrics

if __name__ == "__main__":
    args = parse_args()
    # test_model(args)
    
    # 如果需要验证模型性能，可以取消下面代码的注释并运行
    validate_model(
        model_path=f"/home/fanchen/ultralytics-main/my_project_head_hand/my_detection_model8/weights/best.pt",
        data_path='/home/fanchen/ultralytics-main/ultralytics/cfg/datasets/open_images_hand_head.yaml',
        device=[0, 1, 2, 3]
    )