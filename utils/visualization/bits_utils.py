import numpy as np
import torch
import math

def bpp_calc(likelihoods, img_size):
    N, _, H, W = img_size
    num_pixels = N * H * W
    lbpp = sum(
        (torch.log(l).sum() / (-math.log(2) * num_pixels))
        for l in likelihoods.values()
    ).reshape(1)
    lbpp = lbpp.detach().cpu().item()
    return lbpp

def bitmap(likelihood):
    bit_map = np.transpose(likelihood.detach().cpu().numpy(), (2,3,1,0))[:,:,:,0]
    bit_map = np.sum(-np.log(bit_map)/np.log(2), axis = 2)
    return bit_map