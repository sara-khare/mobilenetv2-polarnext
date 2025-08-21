# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# Force override, even if wrapped
def force_set_annfile(ds_cfg, new_path):
    if isinstance(ds_cfg, dict):
        if 'ann_file' in ds_cfg:
            ds_cfg['ann_file'] = new_path
        if 'dataset' in ds_cfg:
            force_set_annfile(ds_cfg['dataset'], new_path)


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config

    val_ann = '/home/khare/dataset/coco_reduced/annotations/instances_val2017_remap_clean_subset500.json'
    
    cfg = Config.fromfile(args.config)
    # Ensure the val dataloader uses the reduced ann file (safe for nested dicts)
    def _set_annfile_in_dataloader(dataloader_cfg, ann_path):
        if isinstance(dataloader_cfg, dict):
            if 'ann_file' in dataloader_cfg:
                dataloader_cfg['ann_file'] = ann_path
            # sometimes it's nested under 'dataset'
            if 'dataset' in dataloader_cfg:
                _set_annfile_in_dataloader(dataloader_cfg['dataset'], ann_path)

    _set_annfile_in_dataloader(cfg.val_dataloader.dataset, val_ann)
    _set_annfile_in_dataloader(cfg.test_dataloader.dataset, val_ann)  # if present

    # Make sure evaluators know the ann file (so CocoMetric initializes COCO API)
    def _set_annfile_in_evaluator(ev_cfg, ann_path):
        if ev_cfg is None:
            return
        if isinstance(ev_cfg, dict):
            # single evaluator specified as dict
            ev_cfg['ann_file'] = ann_path
        elif isinstance(ev_cfg, (list, tuple)):
            # list of evaluators
            for sub in ev_cfg:
                if isinstance(sub, dict):
                    sub['ann_file'] = ann_path

    # Apply to both val/test evaluator configurations if they exist in cfg
    if cfg.get('val_evaluator', None) is not None:
        _set_annfile_in_evaluator(cfg.val_evaluator, val_ann)
    if cfg.get('test_evaluator', None) is not None:
        _set_annfile_in_evaluator(cfg.test_evaluator, val_ann)

    cfg.val_dataloader.dataset.ann_file = "/home/khare/dataset/coco_reduced/annotations/instances_val2017_remap_clean_subset500.json"
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        
    def _ensure_outfile_prefix_for_eval(evaluator_cfg):
        if isinstance(evaluator_cfg, dict):
            if evaluator_cfg.get('format_only', False) and not evaluator_cfg.get('outfile_prefix'):
                evaluator_cfg['outfile_prefix'] = osp.join(cfg.work_dir, 'results')
        elif isinstance(evaluator_cfg, (list, tuple)):
            for ev in evaluator_cfg:
                _ensure_outfile_prefix_for_eval(ev)

    if cfg.get('test_evaluator', None) is not None:
        _ensure_outfile_prefix_for_eval(cfg.test_evaluator)
    if cfg.get('val_evaluator', None) is not None:
        _ensure_outfile_prefix_for_eval(cfg.val_evaluator)

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']

            force_set_annfile(cfg.val_dataloader.dataset,"/home/khare/dataset/coco_reduced/annotations/instances_val2017_remap_clean_subset500.json")
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
