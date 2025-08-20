# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler
from mmdet.datasets.uavid_dataset import UAVidDataset

from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import LoadAnnotations, PackDetInputs, RandomFlip, Resize
from mmdet.evaluation import CocoMetric

# -------------------------------
# Dataset settings
# -------------------------------
dataset_type = 'UAVidDataset'
data_root = '/home/khare/dataset/UAVid'
backend_args = None

# -------------------------------
# Pipelines
# -------------------------------
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),  # Change to with_mask=True if doing segmentation
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # For test images without annotations, you can comment out LoadAnnotations
    # dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# -------------------------------
# DataLoaders
# -------------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='uavid_train_coco.json',
        data_prefix=dict(img='uavid_train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='uavid_val_coco.json',
        data_prefix=dict(img='uavid_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# For inference on test images without labels
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=None,  # No annotations
        data_prefix=dict(img='uavid_test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# -------------------------------
# Evaluators
# -------------------------------
val_evaluator = dict(
    type=CocoMetric,
    ann_file=f'{data_root}/uavid_val_coco.json',
    metric='bbox',       # Use 'segm' if segmentation masks are available
    format_only=False,
    backend_args=backend_args
)

test_evaluator = dict(
    type=CocoMetric,
    ann_file=None,       # No ground truth for test
    metric='bbox',       # Use 'segm' if segmentation masks are available
    format_only=True,
    backend_args=backend_args
)
