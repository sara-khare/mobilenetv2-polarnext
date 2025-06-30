import torch
import torch.nn as nn
from torch.autograd import Function
from mmdet.registry import MODELS

import native_rasterizer

MODE_BOUNDARY = "boundary"
MODE_MASK = "mask"
MODE_HARD_MASK = "hard_mask"

MODE_MAPPING = {
    MODE_BOUNDARY: 0,
    MODE_MASK: 1,
    MODE_HARD_MASK: 2
}


class SoftPolygonFunction(Function):
    @staticmethod
    def forward(ctx, vertices, width, height, inv_smoothness=1.0, mode=MODE_MASK):
        ctx.width = width
        ctx.height = height
        ctx.inv_smoothness = inv_smoothness
        ctx.mode = MODE_MAPPING[mode]

        vertices = vertices.clone()
        ctx.device = vertices.device
        ctx.batch_size, ctx.number_vertices = vertices.shape[:2]
        
        rasterized = torch.FloatTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0.0).to(device=ctx.device)

        contribution_map = torch.IntTensor(
            ctx.batch_size,
            ctx.height,
            ctx.width).fill_(0).to(device=ctx.device)
        rasterized, contribution_map = native_rasterizer.forward_rasterize(vertices, rasterized, contribution_map, width, height, inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_vertices = torch.FloatTensor(
            ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = native_rasterizer.backward_rasterize(
            vertices, rasterized, contribution_map, grad_output, grad_vertices, ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None


@MODELS.register_module()
class SoftPolygonCUDA(nn.Module):
    MODES = [MODE_BOUNDARY, MODE_MASK, MODE_HARD_MASK]
    
    def __init__(self, width=64, height=64, inv_smoothness=0.1, mode="mask"):
        super(SoftPolygonCUDA, self).__init__()

        self.width = width
        self.height = height
        self.inv_smoothness = inv_smoothness

        if not (mode in SoftPolygonCUDA.MODES):
            raise ValueError("invalid mode: {0}".format(mode))
            
        self.mode = mode

    def forward(self, vertices):
        return SoftPolygonFunction.apply(vertices, self.width, self.height, self.inv_smoothness, self.mode)
