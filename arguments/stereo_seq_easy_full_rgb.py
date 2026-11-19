_base_ = "./s3po/stereo_seq_easy.py"

ModelParams = dict(mask_mode="depth_only")

OptimizationParams = dict(
    iterations=8000,
    position_lr_max_steps=8000,
    densify_until_iter=3000,
    densify_grad_threshold_fine_init=0.00008,
    densify_grad_threshold_after=0.00004,
)
