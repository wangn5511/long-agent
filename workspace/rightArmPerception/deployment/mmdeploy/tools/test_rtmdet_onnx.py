import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test mmdeploy RTMDet ONNX model with an input image")
    parser.add_argument(
        "--deploy-cfg",
        type=str,
        default="configs/mmdet/detection/detection_onnxruntime_static.py",
        help="Path to mmdeploy deploy config")
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="/home/shiyinan/code/scripts/mmpose_demo/mmpose/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py",
        help="Path to original mmdet model config")
    parser.add_argument(
        "--onnx",
        type=str,
        default="mmdeploy_models/mmdet/rtmdet_nano_320/end2end.onnx",
        help="Path to exported end2end.onnx")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path")
    parser.add_argument(
        "--out",
        type=str,
        default="rtmdet_onnx_result.jpg",
        help="Output image path")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for backend model, e.g. "cpu" or "cuda:0"')
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for visualization")
    return parser.parse_args()


def main():
    args = parse_args()

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    backend_model = [args.onnx]  # mmdeploy 接口要求 list
    device = args.device
    image_path = args.image
    out_path = args.out
    score_thr = args.score_thr

    assert os.path.isfile(
        deploy_cfg_path), f"deploy config not found: {deploy_cfg_path}"
    assert os.path.isfile(
        model_cfg_path), f"model config not found: {model_cfg_path}"
    assert os.path.isfile(backend_model[0]), f"ONNX not found: {backend_model[0]}"
    assert os.path.isfile(image_path), f"image not found: {image_path}"

    # 1. 读 config
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    import pdb; pdb.set_trace()

    # 2. 构建 task_processor 和 backend model（会用 onnxruntime 加载 end2end.onnx）
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    model = task_processor.build_backend_model(backend_model)

    # 3. 处理输入图像（做和训练时一样的 resize / norm / pad）
    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(image_path, input_shape)

    # 4. 推理
    with torch.no_grad():
        results = model.test_step(model_inputs)

    # results 是一个 list，每个元素通常是 DetDataSample
    data_sample = results[0]
    pred_instances = data_sample.pred_instances

    # 5. 解析检测结果：bboxes (N,4), scores (N,), labels (N,)
    bboxes = pred_instances.bboxes.cpu().numpy()  # (N, 4)
    scores = pred_instances.scores.cpu().numpy()  # (N,)
    labels = pred_instances.labels.cpu().numpy()  # (N,)

    print(f"Detected {len(bboxes)} instances")
    for i in range(len(bboxes)):
        print(
            f"[{i}] label={int(labels[i])}, score={float(scores[i]):.4f}, box={bboxes[i].tolist()}"
        )

    # 6. 用 OpenCV 画框并保存
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for box, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)
        # 做一下边界裁剪以防出界
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # 画矩形框和文本（绿色框）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{int(label)}:{score:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_path, img)
    print(f"Saved visualization to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
