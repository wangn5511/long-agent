# 部署模块
本模块用于右臂上的模型部署，并完全独立于其他模块。

## mmdeploy
包含 mmdeploy 官方代码库的 clone 版本（版本 1.3.1），其中做了细微的改动，以配合所有右臂产品的模型代码部署工作。

主要功能：
- pytorch 模型到 onnx 模型的转换流程
- onnx 模型到 rknn 的转换流程

### 模型部署流程
#### 1. RTMDet 模型 (手部检测模型)
这是一个开源的手部检测模型框架，适合部署在3588上。转换RKNN的流程如下：
> 参考 https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md 实现
```bash
# 准备模型文件，以开源为例：
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth

# 准备模型配置文件（例子为mmpose代码库 configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py）
# 目前已放在代码库路径下：configs_models/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py
ls configs_models/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py

# 转换 onnx 文件
# NOTE: 由于 rknn 无法接受NMS等后处理算子，因此导出ONNX模型时，需要排除后处理算子。
python tools/deploy.py configs/mmdet/detection/detection_onnxruntime_static_wo_nms.py  \
    configs_models/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py  \
    /home/shiyinan/code/scripts/mmpose_demo/pretrained/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth  \
    demo/resources/det.jpg  \
    --work-dir mmdeploy_models/mmdet/rtmdet_nano_320-8xb32_hand_wo_nms_double_check  \
    --device cpu  \
    --skip-visualize

# 验证ONNX模型
python tools/infer_rtmdet_onnx.py  \
    --onnx mmdeploy_models/mmdet/rtmdet_nano_320-8xb32_hand_wo_nms_double_check/end2end.onnx  \
    --img demo/resources/hand.jpg  \
    --out mmdeploy_models/mmdet/rtmdet_nano_320-8xb32_hand_wo_nms_double_check/hand_demo.jpg  \
    --manual_nms

# 转换ONNX 模型 到 RKNN 模型
# 这是转换为fp16，并未转换到INT8.
python tools/export_rtmdet_rknn_custom.py  \
    mmdeploy_models/mmdet/rtmdet_nano_320-8xb32_hand_wo_nms_double_check/end2end.onnx  \
    --output mmdeploy_models/mmdet/rtmdet_nano_320-8xb32_hand_wo_nms_double_check/end2end.rknn  \
    --target rk3588  \
    --optimization-level 3

# 验证RKNN模型
python tools/infer_rtmdet_rknn.py xxxx  # 可以在3588 芯片上自行推理
```

#### 2. RTMPose 模型 (手部关键点识别模型)
同样开源的手部关键点识别模型，适合部署3588。转换RKNN的流程如下：
> 参考 https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md 实现
```bash
# 准备模型文件，以开源为例：
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth

# 准备模型配置文件。目前已放在代码库路径下：configs_models/mmpose_cfg/rtmpose-m_8xb256-210e_hand5-256x256.py
ls configs_models/mmpose_cfg/rtmpose-m_8xb256-210e_hand5-256x256.py

# 直接转换 rknn 文件（转换onnx后转换rknn，会报错，TODO解决）
python tools/deploy.py  \
    configs/mmpose/pose-detection_simcc_rknn-fp16_static-256x256.py  \
    configs_models/mmpose_cfg/rtmpose-m_8xb256-210e_hand5-256x256.py  \
    /home/shiyinan/code/scripts/mmpose_demo/pretrained/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth  \
    demo/resources/hand.jpg  \
    --work-dir mmdeploy_models/mmpose/rtmpose-m_simcc-hand5_double_check  \
    --device cpu  \
    --skip-visualize

# 验证RKNN 模型 
python tools/infer_rtm_hand_landmarks.py xxx  # 可以在3588 芯片上自行推理
```

#### 3. TSM 视频分类模型 (动态手势识别模型)
> 参考 https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmaction2.md 实现
```bash
# 1. 准备模型pth文件

# 2. 转换ONNX
python tools/deploy.py \
    configs/mmaction/video-recognition/video-recognition_2d_dynamic.py \
    <模型cfg文件> \
    <模型pth文件> \
    tests/data/arm_wrestling.mp4 \
    --work-dir mmdeploy_models/mmaction/tsm/onnx/ \
    --device cpu \
    --skip-visualize
```
