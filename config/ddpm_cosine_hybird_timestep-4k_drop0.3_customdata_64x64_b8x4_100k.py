_base_ = [
    './ddpm_64x64.py',
    './imagenet_noaug_64.py', './default_runtime.py'
]

# set dropout prob as 0.3
model = dict(denoising=dict(dropout=0.3))

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['real_imgs', 'x_0_pred', 'x_t', 'x_t_1'],
        padding=1,
        interval=1000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('denoising_ema'),
        interval=1,
        start_iter=0,
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH')
]

# do not evaluation in training process because evaluation take too much time.
evaluation = None

total_iters = 10000  # 100k
data = dict(samples_per_gpu=4)  # 8x4=32

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

inception_pkl = './work_dirs/inception_pkl/imagenet_64x64.pkl'
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        bgr2rgb=True,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN')))
