"""
Aug 16, 2024.
Optimal Transport method to tighten the feature space
"""
import torch
import ot
import torch.nn.functional as F
import random
import numpy as np


def TOT(features, gt,  d_size=8, sample_size = 200):
    """
    Features: a certain layer's features
    gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
    """
    f_n, f_c, f_h, f_w = features.size()

    features = F.interpolate(features, size=[f_h // d_size, f_w // d_size], mode='nearest')
    features = features.permute(0, 2, 3, 1)  # n, h, w, c
    features = torch.flatten(features, start_dim=0, end_dim=2)

    gt = F.interpolate(gt, size=[f_h // d_size, f_w // d_size], mode='nearest')

    gt = gt.view(-1)   
    _mask = gt > 0.001     # _mask: In case values of some pixels do not exist. For depth estimation, there are some pixels that lack the ground truth values
    _mask = _mask.to(torch.bool)   
    gt = gt[_mask]
    features = features[_mask, :]


    gt = torch.unsqueeze(gt, dim=1)

    tmp = torch.pow(features, 2)
    tmp = torch.sum(tmp, dim=1)
    tmp = torch.sqrt(tmp)
    features = features / torch.mean(tmp)

    # gt = gt / torch.max(gt)

    f_n, f_c = features.size()
    random_indices = np.random.choice(f_n, size=sample_size, replace=False)
    features = features[random_indices, :]
    gt = gt[random_indices, :]
    uniform_dist = torch.full((sample_size,), 1 / sample_size)
    uniform_dist = uniform_dist.cuda()

    C_z = euclidean_dist(features, features)
    addition = torch.eye(C_z.size(0)) * 999
    addition = addition.cuda()
    C_z = C_z + addition
    C_y = euclidean_dist(gt, gt)
    C_y = C_y / torch.max(C_y)
    C_y = C_y + addition
    T_y = ot.sinkhorn(uniform_dist, uniform_dist, C_y, 0.1)  # You can change 0.1 to a smaller value for better performance, yet a too-small value will easily result in NaN
    T_z = ot.sinkhorn(uniform_dist, uniform_dist, C_z, 0.1)

    loss = torch.sum(C_z * T_y) - torch.sum(C_z *T_z)
    loss = torch.abs(loss)      # Please set a large weight for the loss (e.g., 100 in Age-DB for age estimation)
    return loss



def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]
