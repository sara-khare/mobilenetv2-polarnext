from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class UAVidDataset(CocoDataset):
    METAINFO = {
        'classes': ('building', 'clutter', 'tree', 'low vegetation', 'car', 'pavement', 'road'),
        'palette': [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 128, 128)
        ]
    }
