_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[256, 256], output_names=['simcc_x', 'simcc_y'])

backend_config = dict(
    common_config=dict(
        target_platform='rk3588', 
        sparse_infer=False,
        quantized_algorithm='mmse',
        quantized_method='channel',
        single_core_mode=False,
        remove_reshape=True,
        optimization_level=3),
    input_size_list=[[3, 256, 256]],
    quantization_config=dict(do_quantization=False))
