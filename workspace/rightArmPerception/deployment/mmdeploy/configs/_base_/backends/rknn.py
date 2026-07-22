backend_config = dict(
    type='rknn',
    common_config=dict(
        target_platform='rv1126',  # 'rk3588'
        optimization_level=1),
    quantization_config=dict(
        do_quantization=True,
        dataset=None,
        pre_compile=False,
        # NOTE:
        # - rknn-toolkit2 requires rknn_batch_size in [1, 128] if provided.
        # - historically this value was set to -1 as "unset", which will fail
        #   with rknn-toolkit2 (e.g. 2.3.x). Use 1 as a safe default and
        #   override in your deploy config if you need a larger batch.
        rknn_batch_size=1))
