# This config inherits the model and dataset definitions from the base files,
# and then overrides specific settings for this training run.

_base_ = [
    './models/polarnext_mobilenet_torch.py',
    './datasets/coco.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

dataset_type = 'CocoPolarDataset'
data_root = '/home/khare/dataset/coco_reduced/'

model = dict(bbox_head=dict(num_classes=5))
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

backend_args = None
find_unused_parameters=True

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='PolarLoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PolarResize', scale=(1333, 800), keep_ratio=True),
    dict(type='PolarRandomFlip', prob=0.5),
    dict(type='PolarPackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/khare/dataset/coco_reduced/annotations/instances_train2017_remap_clean.json',
        data_prefix=dict(img='/home/khare/dataset/coco_reduced/train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/khare/dataset/coco_reduced/annotations/instances_val2017.json',
        data_prefix=dict(img='/home/khare/dataset/coco_reduced/val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/khare/dataset/coco_reduced/annotations/instances_val2017_remap_clean_subset500.json',
    metric='segm',      
    format_only=False,
    outfile_prefix='/home/khare/PolarNeXt/work_dirs/polarnext_mobilenet_coco/results',
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=None,       # no ground truth for test
    metric='bbox',
    format_only=False,
    outfile_prefix='/home/khare/PolarNeXt/work_dirs/polarnext_mobilenet_coco/results',
    backend_args=None
)