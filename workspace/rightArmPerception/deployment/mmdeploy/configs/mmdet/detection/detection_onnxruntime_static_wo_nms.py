_base_ = ['../_base_/base_static.py', '../../_base_/backends/onnxruntime.py']

# 不导出 NMS，输出 boxes 和 scores
codebase_config = dict(
    export_postprocess=False  # 关键配置：设为 False 则不导出 NMS
)
