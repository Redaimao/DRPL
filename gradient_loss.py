# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from pylab import *


def LaplaceAlogrithm(image, operator_type,cuda_visible,device):
    assert torch.is_tensor(image) is True

    laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)[np.newaxis,:,:].repeat(3,0)
    if cuda_visible:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).cuda()
    else:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)

    image = image - F.conv2d(image,laplace_operator,padding = 1,stride = 1)
    return image

def gradient_loss_merge(img1,gt,cuda_visible, device): #exclusion loss: to make the gradient between img1 and img2 more non-uniform.
    grad_img1 = LaplaceAlogrithm(img1,'fourfields',cuda_visible,device)
    gt = LaplaceAlogrithm(gt,'fourfields',cuda_visible,device)
    gt.requires_grad_(False)

    g_loss = F.l1_loss(grad_img1,gt,size_average=True,reduce=True)

    return g_loss
