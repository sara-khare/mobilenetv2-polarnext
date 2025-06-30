import cv2
import math
import torch
import numpy as np


def matrix_is_inside_polygon(point_list, polygon_list):

    polygon_roll = torch.roll(polygon_list, -1, 0)
    polygon_list = polygon_list[None, :, :].repeat(point_list.shape[0], 1, 1)           # M × N × 2
    polygon_roll = polygon_roll[None, :, :].repeat(point_list.shape[0], 1, 1)           # M × N × 2
    point_list = point_list[:, None, :].repeat(1, polygon_list.shape[1], 1)             # M × N × 2

    x = point_list[..., 0]
    y = point_list[..., 1]

    xi = polygon_list[..., 0]
    yi = polygon_list[..., 1]

    xj = polygon_roll[..., 0]
    yj = polygon_roll[..., 1]

    intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    inside = torch.sum(intersect, dim=-1)

    return inside % 2 == 1


def matrix_get_n_coordinates(start_point_list, contour_points, ray_num=36):
    radian_angle = torch.deg2rad(torch.arange(0, 360, 360 / ray_num, device=start_point_list.device))   # 36
    direction_vector = torch.stack([torch.cos(radian_angle), torch.sin(radian_angle)], dim=1)           # 36 × 2
    direction_vector = direction_vector[None, :, :].repeat(start_point_list.shape[0], 1, 1)             # M × 36 × 2

    contour_lines = torch.cat((contour_points, torch.roll(contour_points, shifts=-1, dims=0)), dim=1)   # N × 4
    ray_start = start_point_list[:, None, :] + torch.zeros_like(direction_vector)
    ray_end = start_point_list[:, None, :] + direction_vector
    ray_lines = torch.cat((ray_start, ray_end), axis=-1)                                                # M × 36 × 4
    ray_lines = ray_lines[:, :, None, :].repeat(1, 1, contour_lines.shape[0], 1)                        # M × 36 × N × 4
    seg_lines = contour_lines[None, None:, :].repeat(ray_lines.shape[0], ray_lines.shape[1], 1, 1)      # M × 36 × N × 4

    x1, y1, x2, y2 = ray_lines[..., 0], ray_lines[..., 1], ray_lines[..., 2], ray_lines[..., 3]
    x3, y3, x4, y4 = seg_lines[..., 0], seg_lines[..., 1], seg_lines[..., 2], seg_lines[..., 3]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denominator[denominator == 0] = 1e-6

    seg_percent = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
    ray_percent = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator

    ray_percent[(ray_percent < 0) | (seg_percent < 0) | (seg_percent > 1) | (denominator == 1e-6)] = 1e-2

    distances = torch.max(ray_percent, dim=2)[0]                                        # M × 36
    intersections = ray_start + distances.unsqueeze(-1) * direction_vector              # M × 36 × 2

    del seg_percent, ray_percent, ray_lines, seg_lines, x1, y1, x2, y2, x3, y3, x4, y4, denominator

    # # 尝试释放显存
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    return distances, intersections


# test
def distance2mask(points, distances, angles=None, num_rays=36, max_shape=None):
    """Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distances (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        num_rays (int):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    """
    if angles is None:
        angles = torch.arange(0, 360, 360 / num_rays, device=distances.device) / 180 * math.pi

    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, num_rays)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :].repeat(num_points, 1)
    cos = cos[None, :].repeat(num_points, 1)

    x = distances * cos + c_x
    y = distances * sin + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, :, None], y[:, :, None]], dim=2)
    return res


def mask2result(masks, ori_shape):
    """Convert detection results to a list of numpy arrays.

    Args:
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    img_h, img_w = ori_shape
    device = masks.device

    mask_results = torch.zeros((masks.shape[0], img_h, img_w))
    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [masks[i].unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
        im_mask = torch.from_numpy(im_mask).to(dtype=torch.uint8, device=device)
        mask_results[i] = im_mask
    return mask_results
