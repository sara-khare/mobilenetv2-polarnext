import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class UAVidDataset(CustomDataset):
    """UAVid dataset for semantic segmentation."""

    CLASSES = (
        'background', 'building', 'road', 'tree', 'low vegetation',
        'moving car', 'static car', 'human'
    )

    PALETTE = [
        [0, 0, 0],          # background
        [128, 0, 0],        # building
        [128, 64, 128],     # road
        [0, 128, 0],        # tree
        [128, 128, 0],      # low vegetation
        [64, 0, 128],       # moving car
        [64, 64, 0],        # static car
        [192, 128, 128]     # human
    ]

    def __init__(self, **kwargs):
        super(UAVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
