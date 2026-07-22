import cv2
import numpy as np
import onnxruntime as ort
import sys
sys.path.append("/home/fanchen/ultralytics-main")
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes, Keypoints
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_coords, scale_boxes



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



    # def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    #     """
    #     简化版非极大值抑制 (NMS) 实现
        
    #     Args:
    #         prediction: 模型预测结果，形状为 (batch_size, num_dets, 5+num_classes)
    #         conf_thres: 置信度阈值
    #         iou_thres: IoU阈值
    #         max_det: 最大检测数量
            
    #     Returns:
    #         经过NMS后的检测结果
    #     """
    #     assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    #     assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        
    #     # 如果prediction为空，直接返回空数组
    #     if prediction.size == 0:
    #         return np.array([])
        
    #     # 确保prediction是二维数组
    #     if prediction.ndim == 1:
    #         prediction = prediction[None, :]  # 添加批次维度
    #     elif prediction.ndim != 2:
    #         raise ValueError(f"Unsupported prediction shape: {prediction.shape}")
            
    #     # 检查是否有足够的列来提取边界框和置信度
    #     if prediction.shape[1] < 6:
    #         # 至少需要5列：4个边界框坐标 + 1个置信度
    #         print(f"Warning: Prediction has only {prediction.shape[1]} columns, expected at least 5")
    #         return np.array([])
        
    #     # 提取边界框坐标和置信度
    #     # 注意：模型输出的是xywh格式（中心坐标+宽高），需要转换为xyxy格式（左上角+右下角）
    #     try:
    #         box_coords_xywh = prediction[:, :4]  # [cx, cy, w, h] - xywh格式
    #         # 转换xywh到xyxy格式
    #         box_coords = np.copy(box_coords_xywh)
    #         box_coords[:, 0] = box_coords_xywh[:, 0] - box_coords_xywh[:, 2] / 2  # x1 = cx - w/2
    #         box_coords[:, 1] = box_coords_xywh[:, 1] - box_coords_xywh[:, 3] / 2  # y1 = cy - h/2
    #         box_coords[:, 2] = box_coords_xywh[:, 0] + box_coords_xywh[:, 2] / 2  # x2 = cx + w/2
    #         box_coords[:, 3] = box_coords_xywh[:, 1] + box_coords_xywh[:, 3] / 2  # y2 = cy + h/2
    #         confidences = prediction[:, 4]  # 置信度 (第5列)
    #     except IndexError as e:
    #         print(f"Error extracting box coordinates: {e}")
    #         return np.array([])
        
    #     # 过滤低置信度的检测
    #     valid_indices = confidences > conf_thres
    #     if not np.any(valid_indices):
    #         return np.array([])
            
    #     filtered_predictions = prediction[valid_indices]
    #     box_coords = box_coords[valid_indices]
    #     confidences = confidences[valid_indices]
        
    #     # 计算每个框的面积
    #     box_areas = (box_coords[:, 2] - box_coords[:, 0]) * (box_coords[:, 3] - box_coords[:, 1])
        
    #     # 按置信度降序排序
    #     sorted_indices = np.argsort(confidences)[::-1]
    #     sorted_boxes = box_coords[sorted_indices]
    #     sorted_confidences = confidences[sorted_indices]
    #     sorted_areas = box_areas[sorted_indices]
    #     sorted_predictions = filtered_predictions[sorted_indices]
        
    #     # 假设 sorted_indices, sorted_boxes 等变量已经准备好
    #     keep_indices = []

    #     while len(sorted_indices) > 0 and len(keep_indices) < max_det:
    #         # 取出当前置信度最高的框（始终是第一个）
    #         current_idx = 0
    #         keep_indices.append(sorted_indices[current_idx])  # 保留当前框
            
    #         # 计算当前框与其他框的 IoU
    #         ious = self._calculate_iou(
    #             sorted_boxes[current_idx], 
    #             sorted_boxes, 
    #             sorted_areas[current_idx], 
    #             sorted_areas
    #         )
            
    #         # 找出需要抑制的框（IoU 大于阈值）
    #         suppress_indices = np.where(ious > iou_thres)[0]
            
    #         # 构建保留掩码（排除被抑制的框，但保留当前框）
    #         keep_mask = np.ones(len(sorted_indices), dtype=bool)
    #         keep_mask[suppress_indices] = False  # 抑制重叠框
            
    #         # 注意这里不再抑制当前框（不设置 keep_mask[current_idx] = False）
            
    #         # 更新剩余的框
    #         sorted_indices = sorted_indices[keep_mask]
    #         sorted_boxes = sorted_boxes[keep_mask]
    #         sorted_confidences = sorted_confidences[keep_mask]
    #         sorted_areas = sorted_areas[keep_mask]
    #         sorted_predictions = sorted_predictions[keep_mask]
        
            
    #     # 返回经过NMS后的结果
    #     if len(keep_indices) > 0:
    #         return filtered_predictions[keep_indices]  # 或者根据实际需求调整返回结构
    #     else:
    #         return np.array([])


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


def main():
    # 模型路径
    onnx_model_path = "/home/fanchen/ultralytics-main/weights_yolov12n/best.onnx"
    
    # 测试视频路径
    test_video_path = "/home/fanchen/yolo-hand-pose/test_demo2.mp4"
    
    # 初始化推理器
    print(f"Loading ONNX model from {onnx_model_path}...")
    detector = YOLOONNXInferencePose(onnx_model_path, img_size=640)
    
    # 加载PyTorch模型用于比较和获取names
    pytorch_model_path = "/home/fanchen/yolo-hand-pose/model/best.pt"
    print(f"Loading PyTorch model from {pytorch_model_path} for comparison...")
    pytorch_model = YOLO(pytorch_model_path)
    
    # 打开测试视频
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {test_video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 定义输出视频写入器
    output_path = "/home/fanchen/ultralytics-main/results/inference_output_onnx.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video... Output will be saved to {output_path}")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        # ONNX推理
        onnx_detections, letterbox = detector.predict(frame)
        
        # 将ONNX检测结果转换为Ultralytics Results对象
        orig_shape = frame.shape[:2]  # (height, width)
        # 使用PyTorch模型的names字典
        names = {0: 'hand', 1: 'head'}
        # 传递ratio_pad参数进行尺度还原
        onnx_results_obj = create_results_object(onnx_detections[:], orig_shape, frame, test_video_path, names, detector.ratio_pad)
        
        # 使用Ultralytics的绘图功能可视化ONNX结果
        onnx_annotated = onnx_results_obj.plot()
        
        # 写入标注帧到输出视频
        out.write(onnx_annotated)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"Processed {frame_count} frames...")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"Done! Output saved to {output_path}")
    print("ONNX inference completed!")


if __name__ == "__main__":
    main()