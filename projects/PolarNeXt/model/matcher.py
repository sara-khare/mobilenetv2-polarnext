import torch

from torch import nn
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TopCostMatcher(nn.Module):
    def __init__(self,
                 num_sample: int = 9,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 weight_class: float = 1.0,
                 weight_mask: float = 1.0,
                 weight_miou: float = 1.0):
        super().__init__()
        self.num_sample = num_sample
        self.alpha = alpha
        self.gamma = gamma
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_miou = weight_miou

    @torch.no_grad()
    def forward(self, label_targs, label_preds, poly_targs, poly_preds, mask_targs, mask_preds, inside_indices):
        device = label_preds.device

        matched_pos_indices = []
        matched_class_targs_list = []
        for label_targ, label_pred, poly_targ, poly_pred, mask_targ, mask_pred, inside_ind in \
                zip(label_targs, label_preds, poly_targs, poly_preds, mask_targs, mask_preds, inside_indices):
            num_points = label_pred.shape[0]
            num_gts = label_targ.shape[0]

            if label_targ.shape[0] == 0:
                matched_pos_indices.append(torch.full((num_points, ), 80, dtype=label_pred.dtype, device=device))
                matched_class_targs_list.append(torch.zeros((0, ), dtype=torch.int64, device=device))
                continue

            label_pred = label_pred.sigmoid()

            neg_cost_class = (1 - self.alpha) * (label_pred ** self.gamma) * (-(1 - label_pred + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - label_pred) ** self.gamma) * (-(label_pred + 1e-8).log())
            cost_class = pos_cost_class[:, label_targ] - neg_cost_class[:, label_targ]

            del neg_cost_class, pos_cost_class, label_pred

            cost_mask = torch.full_like(cost_class, 1e6, device=device)
            total = torch.stack([poly_pred, poly_targ], -1)
            l_max = total.max(dim=2)[0]
            l_min = total.min(dim=2)[0]
            cost_mask[inside_ind] = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()

            del total, l_max, l_min

            cost_miou = torch.full_like(cost_class, 1e6, device=device)
            mask_pred = mask_pred.flatten(1)
            mask_targ = mask_targ.flatten(1)
            a = torch.sum(mask_pred * mask_targ, dim=1)
            b = torch.sum(mask_pred, dim=1)
            c = torch.sum(mask_targ, dim=1)
            dice = (2. * a + 1.0) / (b + c + 1.0)
            cost_miou[inside_ind] = 1 - dice

            del a, b, c, dice, mask_targ, mask_pred

            C = self.weight_class * cost_class + self.weight_mask * cost_mask + self.weight_miou * cost_miou

            top_cost, pred_ind = torch.topk(C, self.num_sample, dim=0, largest=False, sorted=True)

            del C, cost_class, cost_mask, cost_miou

            gt_ind = torch.arange(label_targ.shape[0], dtype=pred_ind.dtype, device=device)
            gt_ind = gt_ind[None, :].repeat(self.num_sample, 1)

            unignored_index = torch.nonzero(top_cost < 1e6, as_tuple=True)
            flatten_pred_ind = pred_ind[unignored_index]
            flatten_gt_ind = gt_ind[unignored_index]

            del unignored_index, top_cost

            pos_class_targ = torch.full((num_points, ), 80, dtype=label_targ.dtype, device=device)
            pos_class_targ[flatten_pred_ind] = label_targ[flatten_gt_ind]
            matched_class_targs_list.append(pos_class_targ)

            del pos_class_targ

            inside_states = torch.full((num_points, num_gts), -1, dtype=pred_ind.dtype, device=device)
            inside_states[inside_ind] = torch.arange(poly_pred.shape[0], device=device)
            pos_indices = inside_states[flatten_pred_ind, flatten_gt_ind]
            matched_pos_indices.append(pos_indices)

            del inside_states

        return matched_class_targs_list, matched_pos_indices


