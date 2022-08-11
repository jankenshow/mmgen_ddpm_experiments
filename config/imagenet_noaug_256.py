# dataset settings
dataset_type = 'mmcls.CustomDataset'

# different from mmcls, we adopt the setting used in BigGAN.
# We use `RandomCropLongEdge` in training and `CenterCropLongEdge` in testing.
# Importantly, the `to_rgb` is set to `False` to remain image orders as BGR.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/trimmed_shibakoppe/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/trimmed_shibakoppe/val',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file=None,
        pipeline=test_pipeline))
