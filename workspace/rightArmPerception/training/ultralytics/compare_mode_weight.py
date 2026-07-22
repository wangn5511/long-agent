# 比较两个模型的权重差异
import torch
def compare_model_weights(model1, model2):
    """
    比较两个模型的权重差异，输出每个层的权重差异统计信息。
    
    Args:
        model1: 第一个模型（torch.nn.Module）
        model2: 第二个模型（torch.nn.Module）
    """
    # model1中的key必在model2中存在
    for name1, param1 in model1.items():
        if name1 not in model2:
            print(f"警告: 模型1中存在参数 {name1}，但模型2中不存在")
            continue
        param2 = model2[name1]
        if param1.shape != param2.shape:
            print(f"警告: 模型参数形状不匹配: {name1} ({param1.shape} vs {param2.shape})")
            continue
        
        # 计算权重差异
        weight_diff = param1.data - param2.data
        # 将差异转换为浮点数，避免整数类型导致的mean()错误
        weight_diff = weight_diff.float()
        mean_diff = torch.mean(torch.abs(weight_diff)).item()
        max_diff = torch.max(torch.abs(weight_diff)).item()
        
        print(f"层: {name1} | 平均权重差异: {mean_diff:.6f} | 最大权重差异: {max_diff:.6f}")


if __name__ == "__main__":
    # 示例用法
    model_path_1 = '/home/fanchen/ultralytics-main/runs/pose/yolo12_hand_keypoints15/weights/last.pt'
    model_path_2 = '/home/fanchen/ultralytics-main/my_project_head_hand/my_detection_model4/weights/best.pt'
    
    model1 = torch.load(model_path_1, weights_only=False, map_location='cpu')['ema'].float().state_dict()
    model2 = torch.load(model_path_2, weights_only=False, map_location='cpu')['model'].float().state_dict()
    
    compare_model_weights(model1, model2)