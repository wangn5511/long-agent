_base_ = ['../_base_/base_static.py', '../../_base_/backends/rknn.py']

# Static shape export for RKNN (RK3588) without exporting ONNX NMS.
# RKNN-Toolkit2 does not support ONNX `NonMaxSuppression` well and may report:
#   "The input 0 of NonMaxSuppression need to be constant ... Remove ... manually"
#
# This config partitions the graph to end at `multiclass_nms:input`, i.e. it
# exports decoded `boxes` + `scores` (right before NMS). Do NMS on CPU side.

onnx_config = dict(input_shape=[320, 320])

codebase_config = dict(model_type='rknn')

backend_config = dict(
    common_config=dict(target_platform='rk3588', optimization_level=1),
    input_size_list=[[3, 320, 320]],
    quantization_config=dict(do_quantization=False, rknn_batch_size=1))

# rtmdet for rknn-toolkit and rknn-toolkit2
# partition_config = dict(
#     type='rknn',  # the partition policy name
#     apply_marks=True,  # should always be set to True
#     partition_cfg=[
#         dict(
#             save_file='model.onnx',  # name to save the partitioned onnx
#             start=['detector_forward:input'],  # [mark_name:input, ...]
#             end=['multiclass_nms:input'],  # [mark_name:input, ...]
#             # NMS inputs in mmdeploy are `boxes` and `scores`
#             output_names=['boxes', 'scores'])
#     ])


