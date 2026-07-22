import cv2
import numpy as np
import onnxruntime as ort
import sys
sys.path.append(fr"E:\实习记录\fc_test\ultralytics-main")

from ultralytics.engine.results import Results
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes, scale_coords





class YOLOONNXInferencePose:
    def __init__(self, onnx_model_path, img_size=640):
        """
        初始化ONNX推理器，确保预处理和后处理与PyTorch版本一致
        
        Args:
            onnx_model_path: ONNX模型路径
            img_size: 输入图像尺寸
        """
        # 初始化ONNX Runtime会话
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        
        # 获取输入信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 设置参数
        self.img_size = img_size
        
        print(f"ONNX模型已加载，输入形状: {self.input_shape}")
    
    def preprocess(self, img):
        """
        图像预处理，与PyTorch版本保持一致
        使用Ultralytics的LetterBox类进行图像预处理
        """
        # 保存原始图像尺寸
        self.orig_shape = img.shape[:2]
        
        # 使用LetterBox进行预处理
        letterbox = LetterBox(new_shape=(self.img_size, self.img_size), auto=False, scaleup=True)
        processed_img = letterbox(image=img)
        
        # 手动计算ratio_pad参数用于后续的尺度还原
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.img_size, self.img_size)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r  # width, height ratios
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        # 计算padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        
        # 保存ratio_pad参数 (ratio, (pad_w, pad_h))
        self.ratio_pad = (ratio, (left, top))
        
        # 转换为RGB
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        processed_img = processed_img.astype(np.float32) / 255.0
        
        # 调整通道顺序为NCHW
        processed_img = processed_img.transpose(2, 0, 1)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        return processed_img, letterbox
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        自定义非极大值抑制 (NMS) 实现，用于处理(8400, 6)维度的预测结果
        
        Args:
            prediction: 模型预测结果，形状为 (num_dets, 6)，格式为 [x, y, w, h, conf_cls1, conf_cls2]
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            max_det: 最大检测数量
            
        Returns:
            经过NMS后的检测结果，形状为 [num_keep, 6]，格式为 [x1, y1, x2, y2, confidence, class_id]
        """
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        
        # 如果prediction为空，直接返回空数组
        if prediction.size == 0:
            return np.array([])
        
        # 确保prediction是二维数组
        if prediction.ndim == 1:
            prediction = prediction[None, :]  # 添加批次维度
        elif prediction.ndim != 2:
            raise ValueError(f"Unsupported prediction shape: {prediction.shape}")
            
        # 检查是否有足够的列来提取边界框和类别置信度
        if prediction.shape[1] < 6:
            raise ValueError(f"Prediction has only {prediction.shape[1]} columns, expected at least 6")
        
        # 提取边界框坐标 (xywh格式)
        box_coords_xywh = prediction[:, :4]  # [x, y, w, h] - xywh格式
        
        # 转换xywh到xyxy格式
        box_coords = np.copy(box_coords_xywh)
        box_coords[:, 0] = box_coords_xywh[:, 0] - box_coords_xywh[:, 2] / 2  # x1 = cx - w/2
        box_coords[:, 1] = box_coords_xywh[:, 1] - box_coords_xywh[:, 3] / 2  # y1 = cy - h/2
        box_coords[:, 2] = box_coords_xywh[:, 0] + box_coords_xywh[:, 2] / 2  # x2 = cx + w/2
        box_coords[:, 3] = box_coords_xywh[:, 1] + box_coords_xywh[:, 3] / 2  # y2 = cy + h/2
        
        # 提取两个类别的置信度并计算最大值和类别ID
        class_confidences = prediction[:, 4:6]  # [conf_cls1, conf_cls2]
        max_confidences = np.max(class_confidences, axis=1)  # 取两个类别的最大置信度
        class_ids = np.argmax(class_confidences, axis=1)  # 取最大置信度对应的类别ID (0或1)
        
        # 过滤低置信度的检测
        valid_indices = max_confidences > conf_thres
        if not np.any(valid_indices):
            return np.array([])
            
        filtered_boxes = box_coords[valid_indices]
        filtered_confidences = max_confidences[valid_indices]
        filtered_class_ids = class_ids[valid_indices]
        filtered_predictions = prediction[valid_indices]
        
        # 计算每个框的面积
        box_areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])
        
        # 按置信度降序排序
        sorted_indices = np.argsort(filtered_confidences)[::-1]
        sorted_boxes = filtered_boxes[sorted_indices]
        sorted_confidences = filtered_confidences[sorted_indices]
        sorted_class_ids = filtered_class_ids[sorted_indices]
        sorted_areas = box_areas[sorted_indices]
        
        keep_indices = []
        
        while len(sorted_indices) > 0 and len(keep_indices) < max_det:
            # 取出当前置信度最高的框（始终是第一个）
            current_idx = 0
            keep_indices.append(sorted_indices[current_idx])  # 保留当前框
            
            # 计算当前框与其他框的 IoU
            ious = self._calculate_iou(
                sorted_boxes[current_idx], 
                sorted_boxes, 
                sorted_areas[current_idx], 
                sorted_areas
            )
            
            # 找出需要抑制的框（IoU 大于阈值 并且 属于同一类别）
            suppress_indices = np.where((ious > iou_thres) & (sorted_class_ids == sorted_class_ids[current_idx]))[0]
            
            # 构建保留掩码（排除被抑制的框，但保留当前框）
            keep_mask = np.ones(len(sorted_indices), dtype=bool)
            keep_mask[suppress_indices] = False  # 抑制重叠框
            
            # 更新剩余的框
            sorted_indices = sorted_indices[keep_mask]
            sorted_boxes = sorted_boxes[keep_mask]
            sorted_confidences = sorted_confidences[keep_mask]
            sorted_class_ids = sorted_class_ids[keep_mask]
            sorted_areas = sorted_areas[keep_mask]
        
        # 准备输出结果 [x1, y1, x2, y2, confidence, cls_id]
        results = np.zeros((len(keep_indices), 4+1+1))
        if len(keep_indices) > 0:
            # 获取保留的检测框
            kept_boxes = filtered_boxes[keep_indices]
            kept_confidences = filtered_confidences[keep_indices]
            kept_class_ids = filtered_class_ids[keep_indices]
            
            # 构建输出结果
            results[:, :4] = kept_boxes  # xyxy格式的边界框
            results[:, 4] = kept_confidences  # 最大置信度
            results[:, 5] = kept_class_ids  # 类别ID
            
        
        return results



    def postprocess(self, pred, letterbox):
        """
        后处理，与PyTorch版本保持一致
        """
        # 应用非极大值抑制
        # 注意：ONNX模型中实际上没有包含NMS，所以我们需要手动实现
        if pred is not None and len(pred) > 0:
            # 处理姿态估计的预测结果
            # pred形状为 [1, 68, 1029] 
            # 其中68 = 4(边界框) + 1(置信度) + 21(关键点) * 3(x,y,conf)
            if pred.ndim == 3:
                # 取第一个批次并转置为 [num_detections, features]
                pred = pred[0].T  # 转换为 [1029, 68]
                
            # 应用NMS
            # 这里我们实现一个简化的NMS，使用默认参数：
            # conf_thres = 0.25, iou_thres = 0.45
            pred = self.non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            
        return pred




    def _calculate_iou(self, box, boxes, area, areas):
        """
        计算一个框与多个框之间的IoU
        
        Args:
            box: 单个框 [x1, y1, x2, y2]
            boxes: 多个框 [[x1, y1, x2, y2], ...]
            area: 单个框的面积
            areas: 多个框的面积数组
            
        Returns:
            IoU数组
        """
        # 计算交集坐标
        x1_inter = np.maximum(box[0], boxes[:, 0])
        y1_inter = np.maximum(box[1], boxes[:, 1])
        x2_inter = np.minimum(box[2], boxes[:, 2])
        y2_inter = np.minimum(box[3], boxes[:, 3])
        
        # 计算交集面积
        inter_width = np.maximum(0, x2_inter - x1_inter)
        inter_height = np.maximum(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        # 计算IoU
        union_area = area + areas - inter_area
        iou = inter_area / (union_area + 1e-6)  # 添加小量避免除零
        
        return iou


    def predict(self, img):
        """
        执行推理并返回处理后的结果
        """
        # 预处理
        input_tensor, letterbox = self.preprocess(img)
        
        # 执行推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理
        results = self.postprocess(outputs[0], letterbox)
        
        return results, letterbox

def create_results_object(onnx_detections, orig_shape, orig_img, img_path, names, ratio_pad=None):
    """
    将ONNX检测结果转换为Ultralytics Results对象以便可视化
    并进行尺度还原以匹配原始图像尺寸
    """
    # 确保onnx_detections是二维数组，并处理输入维度(1,68,1029)
    if onnx_detections.ndim == 3:
        # 输入是(1,68,1029)，需要转置为(1029,68)
        onnx_detections = onnx_detections.squeeze(0).T  # 变为(1029,68)
    
    if onnx_detections.ndim == 1:
        onnx_detections = onnx_detections[None, :]
    
    if len(onnx_detections) == 0 or onnx_detections.shape[0] == 0:
        # 返回空的Results对象
        return Results(orig_img=orig_img, path=img_path, names=names)
    
    # 处理边界框和关键点数据
    if onnx_detections.shape[1] >= 5:
        # 提取边界框信息 [x1, y1, x2, y2, class_1_conf, class_2_conf]
        boxes_data = onnx_detections[:, :].copy()
        # 为了兼容Ultralytics Results对象，我们需要添加一个cls列
        # 将conf复制到cls位置，作为类别置信度
        # cls_data = np.zeros((boxes_data.shape[0], 2))  # 创建cls列，默认为0
        # boxes_data = np.hstack([boxes_data, cls_data])  # 现在是[x1, y1, x2, y2, conf, cls]
        
        # 如果提供了ratio_pad，则进行尺度还原
        if ratio_pad is not None:
            # 对边界框进行尺度还原
            # boxes_data的前4列是边界框坐标
            boxes_xyxy = boxes_data[:, :4].copy()
            
            
            # 将边界框坐标还原到原始图像尺寸
            scaled_boxes = scale_boxes((640, 640), boxes_xyxy, orig_shape, ratio_pad=ratio_pad)
            boxes_data[:, :4] = scaled_boxes
        
        # 处理关键点信息（如果有）
        keypoints_obj = None
        kpts_reshaped = None
        # 从第5列开始提取关键点信息，共63列（21个关键点）
        if onnx_detections.shape[1] >= 68:  # 确保有足够的列
            kpts_data = onnx_detections[:, 5:68].copy()  # 提取第5到67列，共63列
            # 重新整理为 (num_dets, num_kpts, 3) 格式，明确指定num_kpts为21
            num_kpts = 21  # 明确指定关键点数量为21
            if num_kpts > 0 and kpts_data.shape[1] == 63:  # 确保有63列数据
                kpts_reshaped = kpts_data.reshape(-1, num_kpts, 3)
                
                # 如果提供了ratio_pad，则进行尺度还原
                if ratio_pad is not None:
                    # 对关键点进行尺度还原
                    for i in range(kpts_reshaped.shape[0]):  # 对每个检测框的关键点进行处理
                        # 提取关键点坐标 (x, y)
                        kpts_coords = kpts_reshaped[i, :, :2].copy()
                        # 将关键点坐标还原到原始图像尺寸
                        scaled_kpts = scale_coords((224, 224), kpts_coords, orig_shape, ratio_pad=ratio_pad)
                        kpts_reshaped[i, :, :2] = scaled_kpts
                
                # keypoints_obj = Keypoints(kpts_reshaped, orig_shape)
        
        # 创建Results对象，参考Ultralytics的实现
        return Results(orig_img=orig_img, path=img_path, names=names, boxes=boxes_data, keypoints=kpts_reshaped)
    else:
        # 如果没有足够的数据，创建一个空的结果对象
        return Results(orig_img=orig_img, path=img_path, names=names)



# 在文件末尾添加单图推理函数

def run_single_image_inference(onnx_model_path, image_path, save_output=True, output_path=None):
    """
    对单个图像进行YOLOv12 ONNX模型推理
    
    Args:
        onnx_model_path: ONNX模型文件路径
        image_path: 待推理的图像文件路径
        save_output: 是否保存输出图像
        output_path: 输出图像保存路径，若为None则自动生成
        
    Returns:
        Results对象，包含检测结果
    """
    # 初始化推理器
    print(f"加载ONNX模型: {onnx_model_path}")
    detector = YOLOONNXInferencePose(onnx_model_path, img_size=224)
    
    # 读取图像
    print(f"读取图像: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 执行推理
    print("执行推理...")
    onnx_detections, letterbox = detector.predict(img)
    
    # 将ONNX检测结果转换为Ultralytics Results对象
    orig_shape = img.shape[:2]  # (height, width)
    names = {0: "hand", 1: "head"}  # 默认类别名称
    results = create_results_object(onnx_detections, orig_shape, img, image_path, names, detector.ratio_pad)
    
    # 可视化结果
    annotated_img = results.plot()
    
    # 保存结果（如果需要）
    if save_output:
        if output_path is None:
            # 自动生成输出路径
            import os
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            output_path = f"/home/fanchen/ultralytics-main/results/{name_without_ext}_detected.jpg"
        
        cv2.imwrite(output_path, annotated_img)
        print(f"检测结果已保存到: {output_path}")
    
    # # 显示结果（可选）
    # cv2.imshow('YOLO Detection Result', annotated_img)
    # print("按任意键关闭显示窗口...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return results


def run_camera_inference(onnx_model_path):
    """
    使用本地摄像头进行实时推理
    
    Args:
        onnx_model_path: ONNX模型文件路径
    """
    # 初始化推理器
    print(f"Loading ONNX model from {onnx_model_path}...")
    detector = YOLOONNXInferencePose(onnx_model_path, img_size=224)
    
    # 打开默认摄像头（通常是0号摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("摄像头已启动，按 'q' 键退出")
    
    # 创建默认的names字典（假设只有一个类别"hand"）
    names = {0: "hand", 1: "head"}
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，则退出循环
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # ONNX推理
        onnx_detections, letterbox = detector.predict(frame)
        
        # 将ONNX检测结果转换为Ultralytics Results对象
        orig_shape = frame.shape[:2]  # (height, width)
        # 传递ratio_pad参数进行尺度还原
        onnx_results_obj = create_results_object(onnx_detections, orig_shape, frame, "camera", names, detector.ratio_pad)
        
        # 使用Ultralytics的绘图功能可视化ONNX结果
        annotated_frame = onnx_results_obj.plot()
        
        # 显示标注后的帧
        cv2.imshow('YOLO Hand Pose Detection - ONNX', annotated_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")


if __name__ == "__main__":
    # 设置ONNX模型路径
    onnx_model_path = fr"/home/fanchen/ultralytics-main/weights_yolov12n/best_224.onnx"
    
    # 如果您在Windows上运行，可能需要修改路径为Windows格式
    # onnx_model_path = "E:\\实习记录\\fc_test\\yolo-hand-pose\\model\\best.onnx"
    
    # # 运行摄像头推理
    # run_camera_inference(onnx_model_path)

    # 运行单张图像推理
    image_path = "/home/datasets/gesture_dataset/hand_keypoints/images/val/IMG_00001840.jpg"
    run_single_image_inference(onnx_model_path, image_path)
