import copy
import logging
import math
import torch

import numpy as np
import torch.nn as nn

from mmcv.ops import batched_nms
from mmcv.cnn import Scale, ConvModule
from mmengine import print_log
from mmengine.structures import InstanceData

from mmengine.config import ConfigDict
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmdet.models.utils import multi_apply, select_single_mlvl, filter_scores_and_topk
from mmdet.models import AnchorFreeHead
from mmdet.structures.mask import BitmapMasks

from .structure.coder import matrix_get_n_coordinates, distance2mask, mask2result

INF = 1e8


@MODELS.register_module()
class PolarNeXtHead(AnchorFreeHead):
    def __init__(self,
                 num_rays: int = 36,
                 num_sample: int = 9,
                 num_classes: int = 5,
                 in_channels: int = 256,
                 mask_size: Tuple = (64, 64),
                 align_offset: float = 0.5,
                 sampling_radius: float = 1.5,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     loss_weight=1.0),
                 loss_mask: ConfigType = dict(
                     type='PolarIoULoss',
                     loss_weight=1.0),
                 loss_miou=dict(
                     type='RMaskIoULoss',
                     loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 assigner: ConfigType = dict(type='TopCostMatcher'),
                 aligner: ConfigType = dict(
                     type='SoftPolygonCUDA', inv_smoothness=0.1),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.num_rays = num_rays
        self.num_sample = num_sample
        self.mask_size = mask_size
        self.align_offset = align_offset
        self.sampling_radius = sampling_radius
        self.angles = torch.arange(0, 360, 360 / self.num_rays).cuda() / 180 * math.pi

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_miou = MODELS.build(loss_miou)
        self.loss_centerness = MODELS.build(loss_centerness)

        assigner.update(dict(
            alpha=loss_cls.alpha,
            gamma=loss_cls.gamma,
            weight_class=loss_cls.loss_weight,
            weight_mask=loss_mask.loss_weight,
            weight_miou=loss_miou.loss_weight
        ))
        if assigner.type == 'TopCostMatcher':
            assigner.update(dict(num_sample=num_sample))
        self.assigner = TASK_UTILS.build(assigner)

        aligner.update(dict(
                width=mask_size[0],
                height=mask_size[1]))
        self.aligner = MODELS.build(aligner)

    def _init_layers(self) -> None:
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_rays, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        cls_score, poly_pred, _, reg_feat = super().forward_single(x)

        centerness = self.conv_centerness(reg_feat)

        poly_pred = scale(poly_pred).float()
        poly_pred *= stride
        poly_pred = poly_pred.clamp(min=1e-2)

        return cls_score, poly_pred, centerness

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            poly_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert len(cls_scores) == len(poly_preds) == len(centernesses)

        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        centernesses = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]
        poly_preds = [
            poly_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_rays)
            for poly_pred in poly_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        poly_preds = torch.cat(poly_preds, dim=1)
        centernesses = torch.cat(centernesses, dim=1)

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=poly_preds[0].dtype,
            device=poly_preds[0].device)

        gt_labels, poly_preds, poly_targs, mask_preds, mask_targs, inside_indices = self.get_targets(
            points=all_level_points,
            poly_preds=poly_preds,
            batch_gt_instances=batch_gt_instances
        )

        del all_level_points

        centernesses = [
            centerness[inside_ind[0]]
            for centerness, inside_ind in zip(centernesses, inside_indices)
        ]

        loss_dict = self.criterion(
            gt_labels, cls_scores,
            poly_targs, poly_preds,
            mask_targs, mask_preds,
            centernesses, inside_indices
        )
        return loss_dict

    def get_targets(
            self, points: List[Tensor], poly_preds: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[BitmapMasks], List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_points = [center.size(0) for center in points]
        concat_points = torch.cat(points, 0)

        labels_list, poly_preds, poly_targs, mask_preds, mask_targs, inside_states = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            poly_preds,
            points=concat_points,
            num_points_per_lvl=num_points)

        return labels_list, poly_preds, poly_targs, mask_preds, mask_targs, inside_states

    def _get_targets_single(self,
                            gt_instances: InstanceData,
                            poly_preds: Tensor,
                            points: Tensor,
                            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_points = points.size(0)
        num_gts = len(gt_instances)

        gt_labels = gt_instances.labels
        device = gt_labels.device

        gt_bboxes = gt_instances.bboxes
        gt_masks = gt_instances.masks
        gt_polys = gt_instances.polys
        mask_centers = gt_instances.centers

        if num_gts == 0:
            print_log(
                '-----------Warning: polar_target_single get an empty Ground Truth!----------',
                logger='current',
                level=logging.WARNING)
            return gt_labels, \
                   torch.full((0, self.num_rays), 1e-4, dtype=poly_preds, device=device), \
                   torch.full((0, self.num_rays), 1e-4, dtype=poly_preds, device=device), \
                   torch.zeros((0, self.mask_size[0], self.mask_size[1]), dtype=torch.float32, device=device), \
                   torch.zeros((0, self.mask_size[0], self.mask_size[1]), dtype=torch.float32, device=device), \
                   torch.zeros((num_points, num_gts), dtype=torch.bool, device=device)

        gt_bboxes_expend = gt_bboxes[None].expand(num_points, num_gts, 4)
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        xs, ys = points[..., 0], points[..., 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        center_xs = mask_centers[..., 0]
        center_ys = mask_centers[..., 1]
        center_gts = torch.zeros_like(gt_bboxes_expend)
        stride = center_xs.new_zeros(center_xs.shape)

        del mask_centers

        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * self.sampling_radius
            lvl_begin = lvl_end

        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes_expend[..., 0],
                                         x_mins, gt_bboxes_expend[..., 0])
        center_gts[..., 1] = torch.where(y_mins > gt_bboxes_expend[..., 1],
                                         y_mins, gt_bboxes_expend[..., 1])
        center_gts[..., 2] = torch.where(x_maxs > gt_bboxes_expend[..., 2],
                                         gt_bboxes_expend[..., 2], x_maxs)
        center_gts[..., 3] = torch.where(y_maxs > gt_bboxes_expend[..., 3],
                                         gt_bboxes_expend[..., 3], y_maxs)

        del center_xs, center_ys, stride, x_mins, y_mins, x_maxs, y_maxs

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_states = center_bbox.min(-1)[0] > 0

        del xs, ys, center_gts, cb_dist_left, cb_dist_right, cb_dist_top, cb_dist_bottom, center_bbox

        pos_poly_targs = []
        pos_poly_targs_decoded = []
        for i, polygon in enumerate(gt_polys):
            single_poly_targs, single_poly_targs_decoded = matrix_get_n_coordinates(
                points[inside_gt_states[:, i]],
                polygon.to(device),
                ray_num=self.num_rays
            )

            pos_poly_targs.append(single_poly_targs)
            pos_poly_targs_decoded.append(single_poly_targs_decoded)

        inside_indices = torch.nonzero(inside_gt_states.transpose(0, 1), as_tuple=True)
        inside_indices = (inside_indices[1], inside_indices[0])

        points = points[:, None, :].expand(num_points, num_gts, 2)
        poly_preds = poly_preds[:, None, :].expand((num_points, num_gts, self.num_rays))

        pos_points = points[inside_indices]
        pos_poly_preds = poly_preds[inside_indices]
        pos_poly_preds_decoded = distance2mask(pos_points, pos_poly_preds, num_rays=self.num_rays)

        del points, single_poly_targs, single_poly_targs_decoded, inside_gt_states, poly_preds

        pos_bbox_targs_decoded = gt_bboxes[inside_indices[1]]
        pos_bbox_targs_decoded = pos_bbox_targs_decoded.reshape(-1, 2, 2)
        pos_poly_targs = torch.cat(pos_poly_targs, dim=0)
        # pos_poly_targs_decoded = torch.cat(pos_poly_targs_decoded, dim=0)

        union_points = torch.cat([pos_poly_preds_decoded, pos_bbox_targs_decoded], dim=1).detach()
        union_bboxes = torch.stack([
            union_points[..., 0].min(1)[0], union_points[..., 1].min(1)[0],
            union_points[..., 0].max(1)[0], union_points[..., 1].max(1)[0]
        ], dim=-1)

        # del pos_poly_targs_decoded, union_points

        bbox_width = union_bboxes[:, 2] - union_bboxes[:, 0]
        bbox_height = union_bboxes[:, 3] - union_bboxes[:, 1]
        bbox_minx = union_bboxes[:, 0][:, None].expand(-1, self.num_rays)
        bbox_miny = union_bboxes[:, 1][:, None].expand(-1, self.num_rays)
        bbox_width = bbox_width[:, None].expand(-1, self.num_rays)
        bbox_height = bbox_height[:, None].expand(-1, self.num_rays)

        pos_poly_preds_decoded[:, :, 0] = (pos_poly_preds_decoded[:, :, 0] - bbox_minx) / bbox_width * self.mask_size[0] - self.align_offset
        pos_poly_preds_decoded[:, :, 1] = (pos_poly_preds_decoded[:, :, 1] - bbox_miny) / bbox_height * self.mask_size[1] - self.align_offset

        del bbox_width, bbox_height, bbox_minx, bbox_miny

        num_pos = pos_poly_targs.shape[0]
        group_size = 500
        if num_pos > group_size:
            pos_mask_preds = []
            pos_mask_targs = []
            for i in range(0, num_pos, group_size):
                end_idx = min(i + group_size, num_pos)
                pos_mask_preds.append(self.aligner(pos_poly_preds_decoded[i: end_idx]))
                pos_mask_targs.append(
                    gt_masks.crop_and_resize(
                        union_bboxes[i: end_idx],
                        out_shape=self.mask_size,
                        inds=inside_indices[1][i: end_idx],
                        device=device,
                        interpolation='bilinear',
                        binarize=True
                    ).to_tensor(dtype=torch.float32, device=device)
                )
            pos_mask_preds = torch.cat(pos_mask_preds, dim=0)
            pos_mask_targs = torch.cat(pos_mask_targs, dim=0)
        else:
            pos_mask_preds = self.aligner(pos_poly_preds_decoded)
            pos_mask_targs = gt_masks.crop_and_resize(
                union_bboxes,
                out_shape=self.mask_size,
                inds=inside_indices[1],
                device=device,
                interpolation='bilinear',
                binarize=True
            ).to_tensor(dtype=torch.float32, device=device)

        return gt_labels, pos_poly_preds, pos_poly_targs, pos_mask_preds, pos_mask_targs, inside_indices

    def criterion(self,
                  label_targs, label_preds,
                  poly_targs, poly_preds,
                  mask_targs, mask_preds,
                  centernesses, inside_indices):

        label_targs, pos_indices = self.assigner(
            label_targs, label_preds, poly_targs, poly_preds, mask_targs, mask_preds, inside_indices
        )

        # -- temporary sanitation: clamp / drop invalid labels to avoid crash --
        # label_targs is a list of tensors (per image). Convert to list, clamp any >= num_classes.
        sanity_fixed = False
        for ii, lt in enumerate(label_targs):
            if isinstance(lt, torch.Tensor) and lt.numel() > 0:
                if int(lt.max().item()) >= self.num_classes:
                    # replace illegal labels with 0 (or choose an allowed label)
                    # NOTE: This is a *temporary* workaround — it hides the root cause.
                    lt = lt.clone()
                    illegal_mask = lt >= self.num_classes
                    if illegal_mask.any():
                        lt[illegal_mask] = 0  # or choose torch.zeros_like(lt[illegal_mask]) if appropriate
                        label_targs[ii] = lt
                        sanity_fixed = True
        if sanity_fixed:
            print(f"TMP-SANITY: replaced labels >= {self.num_classes} with 0 to avoid crash", flush=True)
        # --------------------------------------------------------------------



        del inside_indices

        poly_targs = torch.cat([
            poly_targ[pos_ind]
            for poly_targ, pos_ind in zip(poly_targs, pos_indices)
        ], dim=0)

        poly_preds = torch.cat([
            poly_pred[pos_ind]
            for poly_pred, pos_ind in zip(poly_preds, pos_indices)
        ], dim=0)

        mask_targs = torch.cat([
            mask_targ[pos_ind]
            for mask_targ, pos_ind in zip(mask_targs, pos_indices)
        ], dim=0)

        mask_preds = torch.cat([
            mask_pred[pos_ind]
            for mask_pred, pos_ind in zip(mask_preds, pos_indices)
        ], dim=0)

        centernesses = torch.cat([
            centerness[pos_ind]
            for centerness, pos_ind in zip(centernesses, pos_indices)
        ], dim=0)

        del pos_indices

        num_pos = poly_targs.shape[0]
        label_preds = label_preds.reshape((-1, self.num_classes))
        label_targs = torch.cat(label_targs, dim=0)
        loss_cls = self.loss_cls(label_preds, label_targs, avg_factor=num_pos)

        del label_preds, label_targs

        if num_pos > 0:
            mask_merged = torch.stack([mask_preds.detach().flatten(1), mask_targs.flatten(1)], dim=-1)
            mask_inter = mask_merged.min(-1)[0].sum(1)
            mask_union = mask_merged.max(-1)[0].sum(1)
            mask_ious = mask_inter / mask_union

            del mask_merged, mask_inter, mask_union

            centerness_denorm = max(reduce_mean(mask_ious.sum()), 1e-6)

            loss_centerness = self.loss_centerness(
                centernesses, mask_ious, avg_factor=num_pos)
            loss_mask = self.loss_mask(
                poly_preds,
                poly_targs,
                weight=mask_ious,
                avg_factor=centerness_denorm)
            loss_miou = self.loss_miou(
                mask_preds,
                mask_targs,
                weight=mask_ious,
                avg_factor=centerness_denorm)
        else:
            loss_mask = poly_preds.sum()
            loss_miou = poly_preds.sum()
            loss_centerness = centernesses.sum()

        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_miou=loss_miou,
            loss_centerness=loss_centerness)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        mask_preds: List[Tensor],
                        centernesses: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        assert len(cls_scores) == len(mask_preds) == len(centernesses)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        # if self.num_sample == 1:
        #     mlvl_priors = torch.cat(mlvl_priors, dim=0)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            mask_pred_list = select_single_mlvl(
                mask_preds, img_id, detach=True)
            centerness_list = select_single_mlvl(
                centernesses, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                mask_pred_list=mask_pred_list,
                centerness_list=centerness_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                mask_pred_list: List[Tensor],
                                centerness_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_mask_preds = []
        mlvl_centerness = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, mask_pred, centerness, priors) in \
                enumerate(zip(cls_score_list, mask_pred_list, centerness_list, mlvl_priors)):
            assert cls_score.size()[-2:] == mask_pred.size()[-2:]

            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.num_rays)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(mask_pred=mask_pred, priors=priors))

            mask_pred = filtered_results['mask_pred']
            priors = filtered_results['priors']
            centerness = centerness[keep_idxs]

            mask_pred = distance2mask(priors, mask_pred, angles=self.angles, num_rays=self.num_rays, max_shape=img_shape)

            mlvl_mask_preds.append(mask_pred)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_centerness.append(centerness)

        results = InstanceData()
        results.masks = torch.cat(mlvl_mask_preds)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.centerness = torch.cat(mlvl_centerness)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            mask_pred = results.masks
            scale_factor = torch.Tensor(scale_factor).to(results.masks.device)
            scale_factor = scale_factor.unsqueeze(0).repeat(self.num_rays, 1)
            scale_factor = scale_factor.unsqueeze(0).repeat(mask_pred.shape[0], 1, 1)
            mask_pred = mask_pred * scale_factor
            results.masks = mask_pred

        centerness = results.pop('centerness')
        results.scores = results.scores * centerness

        if results.masks.numel() > 0:
            bbox_pred = torch.stack([
                mask_pred[..., 0].min(1)[0], mask_pred[..., 1].min(1)[0],
                mask_pred[..., 0].max(1)[0], mask_pred[..., 1].max(1)[0]
            ], dim=-1)

            det_bboxes, keep_idxs = batched_nms(bbox_pred, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
        else:
            results.bboxes = results.scores.new_zeros(len(results.scores), 4)

        mask_pred = mask2result(
            results.masks, img_meta['ori_shape']
        )
        results.masks = mask_pred

        return results

