import sys
sys.path.append('/home/fanchen/ultralytics_RTMO_up')  # 确保ultralytics路径在sys.path中

from datetime import datetime
from ultralytics import __version__
from copy import deepcopy
from ultralytics import YOLO
import torch
import os
import yaml



def save_model_weights(model, save_path):
    """
    保存模型权重为.pt格式，符合Ultralytics标准格式
    
    Args:
        model: 训练好的模型
        save_path: 保存路径
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 如果model是YOLO对象，则获取底层模型
        if hasattr(model, 'model'):
            model_to_save = model.model
        else:
            model_to_save = model
            
        # 创建符合Ultralytics格式的检查点字典
        ckpt = {
            'model': None,  # 保存半精度模型
            'date': datetime.now().isoformat(),
            'version': __version__,
            'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
            'docs': 'https://docs.ultralytics.com',
        }
        
        # 如果模型有yaml属性，添加到检查点
        if hasattr(model_to_save, 'yaml'):
            ckpt['yaml'] = model_to_save.yaml
            
        # 如果模型有args属性，添加到检查点
        if hasattr(model_to_save, 'args'):
            ckpt['train_args'] = model_to_save.args
            
        # 如果模型有names属性，添加到检查点
        if hasattr(model_to_save, 'names'):
            ckpt['names'] = model_to_save.names
            
        # 如果模型有nc属性，添加到检查点
        if hasattr(model_to_save, 'nc'):
            ckpt['nc'] = model_to_save.nc
            
        # 如果模型有stride属性，添加到检查点
        if hasattr(model_to_save, 'stride'):
            ckpt['stride'] = model_to_save.stride
            
        # 如果模型有task属性，添加到检查点
        if hasattr(model_to_save, 'task'):
            ckpt['task'] = model_to_save.task

        # 特别处理关键点形状属性
        if hasattr(model_to_save, 'kpt_shape'):
            ckpt['kpt_shape'] = model_to_save.kpt_shape
        else:
            # 默认设置为[21, 3]，表示21个关键点，每个有x,y,conf三个值
            model_to_save.kpt_shape = [21, 3]
            ckpt['kpt_shape'] = [21, 3]

        if hasattr(model_to_save, 'kpt_names'):
            ckpt['kpt_names'] = model_to_save.kpt_names
        else:
            ckpt['kpt_names'] = {0: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
                                  1: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']}
            model_to_save.kpt_names = ckpt['kpt_names']
        
        ckpt['model'] = deepcopy(model_to_save).half()  # 保存半精度模型
        # 保存模型权重
        torch.save(ckpt, save_path)
        print(f"模型权重已保存为.pt格式: {save_path}")
    except Exception as e:
        print(f"保存模型权重时出错: {e}")
        # 如果上面的方法失败，尝试简化版本
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 创建简化的检查点字典
            ckpt = {
                'model': model.model if hasattr(model, 'model') else model,
                'date': datetime.now().isoformat(),
                'version': __version__,
                'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
                'docs': 'https://docs.ultralytics.com',
                'nc': 1,  # 类别数
                'names': ['hand'],  # 类别名称
            }
            
            # 保存模型权重
            torch.save(ckpt, save_path)
            print(f"模型权重已保存为.pt格式（简化版）: {save_path}")
        except Exception as e2:
            print(f"保存模型权重时再次出错: {e2}")








def main():
    # 1. 配置参数
    YOLO12_POSE_CONFIG = '/home/fanchen/ultralytics_RTMO_up/ultralytics/cfg/models/12/yolo12n-pose_my.yaml'
    DETECTION_WEIGHTS = f"/home/fanchen/ultralytics_RTMO_up/best_hand_head_detection.pt" #'/home/fanchen/ultralytics-main/weights_yolov12n/best.pt'  # 预训练检测模型
    imported_weights_path = '/home/fanchen/ultralytics_RTMO_up/pose_weights_add_detection_www.pt'  # 保存导入权重后的模型路径



    
    # 2. 检查文件和目录是否存在
    if not os.path.exists(DETECTION_WEIGHTS):
        raise FileNotFoundError(f"检测模型权重不存在: {DETECTION_WEIGHTS}")
    
    
    # 4. 首先使用pose配置创建模型结构
    print(f"加载YOLO12-pose模型配置: {YOLO12_POSE_CONFIG}")
    pose_model = YOLO(YOLO12_POSE_CONFIG)
    
    # 5. 然后加载预训练检测模型权重
    print(f"加载预训练检测模型权重: {DETECTION_WEIGHTS}")
    detection_model = YOLO(DETECTION_WEIGHTS)   #(f"/home/fanchen/ultralytics-main/ultralytics/cfg/models/12/yolo12n-head_hand.yaml")  # (DETECTION_WEIGHTS) 

    # 6. 迁移权重：从检测模型到pose模型
    print("开始迁移权重...")
    pose_state_dict = pose_model.state_dict()
    detection_state_dict = detection_model.state_dict()
    
    transferred = 0
    skipped = 0
    
    for key, value in pose_state_dict.items():
        if key in detection_state_dict:
            # 确保形状匹配  
            if detection_state_dict[key].shape == value.shape:
                pose_state_dict[key] = detection_state_dict[key]
                transferred += 1
            else:
                print(f"形状不匹配, {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                if key == "model.model.1.conv.weight" or key == "model.model.3.conv.weight":
                    print(f"注意: 该权重形状不匹配，是因为groups!=1导致的。")

                    if key == "model.model.1.conv.weight":
                        # 由groups=2的权重构造groups=1的权重
                        old_weight = pose_state_dict[key]
                        c_out, c_in, k_h, k_w = old_weight.shape
                        G = 2  # 原始groups数量
                        weight_g2 = detection_state_dict[key]  # [32, 8, 3, 3]
                        weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [32, 16, 3, 3]
                        for g in range(G):
                            in_start = g * (c_in // G)          # 0, 8
                            in_end   = (g + 1) * (c_in // G)    # 8, 16

                            out_start = g * (c_out // G)        # 0, 16
                            out_end   = (g + 1) * (c_out // G)  # 16, 32

                            # 从 weight_g2 中取出第 g 组的权重（连续存储）
                            # 注意：weight_g2 的输出通道是连续的 0~31，所以第 g 组就是 [out_start:out_end]
                            weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g2[out_start:out_end, :, :, :]

                        pose_state_dict[key] = weight_g1
                        transferred += 1
                    elif key == "model.model.3.conv.weight":
                        # 由groups=4的权重构造groups=1的权重
                        old_weight = pose_state_dict[key]
                        c_out, c_in, k_h, k_w = old_weight.shape
                        G = 4  # 原始groups数量
                        weight_g4 = detection_state_dict[key]  # [64, 8, 3, 3]
                        weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [64, 32, 3, 3]
                        for g in range(G):
                            in_start = g * (c_in // G)          # 0, 8, 16, 24
                            in_end   = (g + 1) * (c_in // G)    # 8, 16, 24, 32

                            out_start = g * (c_out // G)        # 0, 16, 32, 48
                            out_end   = (g + 1) * (c_out // G)  # 16, 32, 48, 64

                            # 从 weight_g4 中取出第 g 组的权重（连续存储）
                            weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g4[out_start:out_end, :, :, :]

                        pose_state_dict[key] = weight_g1
                        transferred += 1
                    else:
                        raise ValueError("不应到达此处")
                else:
                    skipped += 1
                    print(f"形状不匹配，跳过权重: {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                        
                
        else:
            print(f"Pose模型中不存在该权重，跳过: {key}")  # model.model.1.conv.weight model.model.3.conv.weight
            skipped += 1

    
    # 加载迁移后的权重
    pose_model.load_state_dict(pose_state_dict)
    print(f"权重迁移完成: 成功迁移 {transferred} 个权重，跳过 {skipped} 个权重")

    # 保存导入权重后的模型
    save_model_weights(pose_model, imported_weights_path)

    
if __name__ == '__main__':
    main()