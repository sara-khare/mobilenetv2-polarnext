# model settings
model = dict(
    type='PolarNeXt',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,        # standard width
        out_indices=(1, 2, 4, 6),   # layers to extract features for FPN
        init_cfg= None
    ),
    neck=dict(
        type='FPN',
        # These are the correct output channels for MobileNetV2 (at out_indices 2, 4, 6, 7)
        in_channels=[24, 32, 96, 320], 
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True
    ),
    bbox_head=dict(
        type='PolarNeXtHead',
        num_rays=36,
        num_sample=9,
        num_classes=5, 
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        assigner=dict(type='TopCostMatcher'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_mask=dict(
            type='PolarIoULoss',
            loss_weight=1.0),
        loss_miou=dict(
            type='RMaskIoULoss',
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)
