# This config inherits the model and dataset definitions from the base files,
# and then overrides specific settings for this training run.

_base_ = [
    './models/polarnext_mobilenet_torch.py',
    './datasets/uavid.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=5))

# This is still needed to correctly import the custom model head
custom_imports = dict(
    imports=['projects.PolarNeXt.model'], allow_failed_imports=False)

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)

# The dataloader is also defined in the `uavid.py` base file, so we can
# just specify the batch size and num_workers here.
train_dataloader = dict(
    batch_size=4,
    num_workers=4
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# Override COCO evaluators with UAVid files
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/khare/dataset/UAVid/uavid_val_coco.json',
    metric='bbox',       # or 'segm' if using segmentation
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=None,       # no ground truth for test
    metric='bbox',
    format_only=True,
    backend_args=None
)