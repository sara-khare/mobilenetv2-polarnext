import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class PolarIoULoss(nn.Module):
    def __init__(self,
                 loss_weight: float = 1.0):
        super(PolarIoULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                ) -> Tensor:

        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        if weight is not None:
            loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss * self.loss_weight


def rmask_iou_loss(
        input,
        target,
        weight=None,
        smooth=1.0,
        reduction='mean',
        naive_dice=True,
        avg_factor=None):

    input = input.flatten(1)
    target = target.flatten(1)

    a = torch.sum(input * target, dim=1)

    if naive_dice:
        b = torch.sum(input, dim=1)
        c = torch.sum(target, dim=1)
        iou = (2. * a + smooth) / (b + c + smooth)
    else:
        b = torch.sum(input * input, dim=1) + smooth
        c = torch.sum(target * target, dim=1) + smooth
        iou = (2 * a) / (b + c)

    loss = 1 - iou
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(input)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class RMaskIoULoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 naive_dice=True,
                 smooth=1.0):

        super(RMaskIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.naive_dice = naive_dice
        self.smooth = smooth

    def forward(self,
                input,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * rmask_iou_loss(
            input,
            target,
            weight,
            smooth=self.smooth,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss
