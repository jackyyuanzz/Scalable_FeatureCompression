# This file contains modules common to various models

import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
from compressai.layers import GDN

import cv2
        
class DepthConv(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s = 1, activation='leakyRelu', **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding= k // 2, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if activation=='leakyRelu':
            self.act = nn.LeakyReLU()
        elif activation=='None':
            self.act = nn.Identity()
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out)
        return out
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.01)
        m.bias.data.fill_(0.001)

class ModulatedC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, proj_dim = 16, mode = 1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ModulatedC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.proj = nn.Sequential(nn.Linear(2*proj_dim, 2*proj_dim), nn.LeakyReLU(), nn.Linear(2*proj_dim, c2))
        self.proj.apply(init_weights)
        self.c2 = c2
        if mode == 1:
            self.mode = 'add'
        elif mode == 2:
            self.mode = 'mult'
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x, beta_projected = None):
        if beta_projected == None:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            if self.mode == 'add':
                return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)) + self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)
            elif self.mode == 'mult':
                return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)) * self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)

class ModulatedConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, proj_dim = 16, mode = 1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ModulatedConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.proj = nn.Sequential(nn.Linear(2*proj_dim, 2*proj_dim), nn.LeakyReLU(), nn.Linear(2*proj_dim, c2))
        self.proj.apply(init_weights)
        self.c2 = c2
        if mode == 1:
            self.mode = 'add'
        elif mode == 2:
            self.mode = 'mult'

    def forward(self, x, beta_projected = None):
        if beta_projected == None:
            return self.act(self.bn(self.conv(x)))
        else:
            if self.mode == 'add':
                return self.act(self.bn(self.conv(x))) + self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)
            elif self.mode == 'mult':
                return self.act(self.bn(self.conv(x))) + self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)

    def fuseforward(self, x, beta_projected = None):
        if beta_projected == None:
            return self.act(self.conv(x))
        else:
            if self.mode == 'add':
                return self.act(self.conv(x)) + self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)
            elif self.mode == 'mult':
                return self.act(self.conv(x)) + self.proj(beta_projected).reshape(x.shape[0], self.c2, 1, 1)

class Conv_ScalableLinear(nn.Module):
    # Standard convolution
    def __init__(self, c1, C2_List, active_scales=1 , k=1, s=1, p=None, g=1, act=False, bn=False, N_freeze=0, bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if active_scales > len(C2_List):
            raise Exception('Active scales cannot be greater than number of convs')
        self.C2_List = C2_List
        
        if not isinstance(s, list):
            stride = [s]*len(C2_List)
        else:
            stride = s
        if not isinstance(k, list):
            kernel_size = [k]*len(C2_List)
        else:
            kernel_size = k
        
        self.CONV = []
        for i, c2 in enumerate(C2_List):
            self.CONV.append(nn.Conv2d(c1, c2, kernel_size[i], stride[i], autopad(kernel_size[i], p), groups=g, bias=bias))
        self.CONV = nn.ModuleList(self.CONV)
        
        for i_conv in range(N_freeze):
            # Freeze the conv layer
            for param in self.CONV[i_conv].parameters():
                param.requires_grad = False
                
        if bn == True:
            self.BN = nn.ModuleList([nn.BatchNorm2d(ch_out) for ch_out in C2_List])
            for i_bn in range(N_freeze):
                # Freeze the BN layer
                for param in self.BN[i_bn].parameters():
                    param.requires_grad = False
        else:
            self.BN = None
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.active_scales = active_scales
        
    def forward(self, x, active=None):
        Out = []
        active_scales = active if active != None else self.active_scales
        # Check: If x is a list of tensor, concatenate them along channel dimension
        if isinstance(x, list):
            x = torch.cat(x, dim = 1)
        for i_conv in range(active_scales):
            if self.BN:
                x_out = self.act(self.BN[i_conv](self.CONV[i_conv](x)))
            else:
                x_out = self.act(self.CONV[i_conv](x))
            Out.append(x_out)
        # Out = torch.cat(Out, dim=1)
        return Out
        
    def clear_grad_scalable(self, clear_layers=0):
        for i_conv in range(clear_layers):
            self.CONV[i_conv].zero_grad()
        if self.BN:
            for i_bn in range(clear_layers):
                self.BN[i_bn].zero_grad()
    
class ConvInv_ScalableLinear(nn.Module):
    # Standard convolution
    def __init__(self, C1_List, c2, active_scales=1 , k=1, s=1, p=None, g=1, act=False, bn=False, N_freeze=0, bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        
        self.c2 = c2
        if active_scales > len(C1_List):
            raise Exception('Active scales cannot be greater than number of convs')
        self.C1_List = C1_List
        
        if not isinstance(s, list):
            stride = [s]*len(C1_List)
        else:
            stride = s
        if not isinstance(k, list):
            kernel_size = [k]*len(C1_List)
        else:
            kernel_size = k
            
        self.CONV = []
        for i, c1 in enumerate(C1_List):
            s = stride[i]
            k = kernel_size[i]
            if s == 1:
                self.CONV.append(nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias))
            elif s == 2:
                self.CONV.append(TransposeConv(c1, c2, k, s))
        self.CONV = nn.ModuleList(self.CONV)
            
        for i_conv in range(N_freeze):
            # Freeze the conv layer
            for param in self.CONV[i_conv].parameters():
                param.requires_grad = False
                
        if bn == True:
            self.BN = nn.ModuleList([nn.BatchNorm2d(c2) for _ in range(len(C1_List))])
            for i_bn in range(N_freeze):
                # Freeze the BN layer
                for param in self.BN[i_bn].parameters():
                    param.requires_grad = False
        else:
            self.BN = None
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.active_scales = active_scales
        # Stride for the first tensor, final output spatial shape should be same for all scalable layers
        # Use stride[0] * first tensor shape to calculate the final output shape
        self.stride = stride[0]
        
        
    def forward(self, x, active = None):
        active_scales = active if active != None else self.active_scales
        
        # if active_scales == 4 and x.shape[1] == 8:
            # x = list(torch.split(x, [8], dim=1))
            # active_scales = active_scales-3
        # elif active_scales == 5:
            # x = list(torch.split(x, [8,8], dim=1))
            # active_scales = active_scales-3
        # elif active_scales == 6:
            # x = list(torch.split(x, [8,8,8], dim=1))
            # active_scales = active_scales-3
        # elif active_scales == 7:
            # x = list(torch.split(x, [8,8,8,8], dim=1))
            # active_scales = active_scales-3
            
        if not isinstance(x, list):
            x = torch.split(x, self.C1_List[0:active_scales], dim=1)
        x_shape = x[0].shape
        x_out = torch.zeros(x_shape[0], self.c2, x_shape[2]*self.stride, x_shape[3]*self.stride, device = x[0].device)
        for i_conv in range(active_scales):
            if self.BN:
                x_out += self.BN[i_conv](self.CONV[i_conv](x[i_conv]))
            else:
                x_out += self.CONV[i_conv](x[i_conv])
        x_out =  self.act(x_out)
        return x_out
        
    def clear_grad_scalable(self, clear_layers=0):
        for i_conv in range(clear_layers):
            self.CONV[i_conv].zero_grad()
        if self.BN:
            for i_bn in range(clear_layers):
                self.BN[i_bn].zero_grad()

class Conv_ScalableLinear_Multi(Conv_ScalableLinear):
    # Standard convolution
    def __init__(self, c1, C2_List, active_scales=1 , k=1, s=1, p=None, g=1, act=False, bn=False, N_freeze=0, N_layers=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, C2_List, active_scales , k, s, p, g, act, bn, N_freeze)
        CONV_LIST = []
        for ch_out in C2_List:
            multi_conv = []
            for i in range(N_layers):
                if i == 0 and N_layers == 1:
                    multi_conv.append(nn.Conv2d(c1, ch_out, k, s, autopad(k, p), groups=g, bias=True))
                if i == 0 and N_layers != 1:
                # First layer
                    multi_conv.append(nn.Conv2d(c1, ch_out, k, s, autopad(k, p), groups=g, bias=True))
                    multi_conv.append(nn.BatchNorm2d(ch_out))
                    multi_conv.append(nn.LeakyReLU())
                elif i == N_layers - 1:
                    # Last layer
                    multi_conv.append(nn.Conv2d(ch_out, ch_out, k, s, autopad(k, p), groups=g, bias=True))
                else:
                    multi_conv.append(nn.Conv2d(ch_out, ch_out, k, s, autopad(k, p), groups=g, bias=True))
                    multi_conv.append(nn.BatchNorm2d(ch_out))
                    multi_conv.append(nn.LeakyReLU())
            CONV_LIST.append(nn.Sequential(*multi_conv))
        self.CONV = nn.ModuleList(CONV_LIST)
        for i_conv in range(N_freeze):
            # Freeze the conv layer
            for param in self.CONV[i_conv].parameters():
                param.requires_grad = False
                
class ConvInv_ScalableLinear_Multi(ConvInv_ScalableLinear):
    # Standard convolution
    def __init__(self, C1_List, c2, active_scales=1 , k=1, s=1, p=None, g=1, act=False, bn=False, N_freeze=0, N_layers=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(C1_List, c2, active_scales , k, s, p, g, act, bn, N_freeze,)
        CONV_LIST = []
        for ch_in in C1_List:
            multi_conv = []
            for i in range(N_layers):
                if i == 0 and N_layers == 1:
                    multi_conv.append(nn.Conv2d(ch_in, c2, k, s, autopad(k, p), groups=g, bias=True))
                elif i == 0 and N_layers != 1:
                # First layer
                    multi_conv.append(nn.Conv2d(ch_in, ch_in, k, s, autopad(k, p), groups=g, bias=True))
                    multi_conv.append(nn.BatchNorm2d(ch_in))
                    multi_conv.append(nn.LeakyReLU())
                elif i == N_layers - 1:
                    # Last layer
                    multi_conv.append(nn.Conv2d(ch_in, c2, k, s, autopad(k, p), groups=g, bias=True))
                else:
                    multi_conv.append(nn.Conv2d(ch_in, ch_in, k, s, autopad(k, p), groups=g, bias=True))
                    multi_conv.append(nn.BatchNorm2d(ch_in))
                    multi_conv.append(nn.LeakyReLU())
            CONV_LIST.append(nn.Sequential(*multi_conv))
        self.CONV = nn.ModuleList(CONV_LIST)
        for i_conv in range(N_freeze):
            # Freeze the conv layer
            for param in self.CONV[i_conv].parameters():
                param.requires_grad = False

class JPEG_Comp(nn.Module):
    # Use JPEG to compress each channel of the input feature map
    def __init__(self, quality = 40, min_val=-1, max_val=1):
        super().__init__()
        self.quality = quality
        self.min_val = min_val
        self.max_val = max_val
        self.dr = self.max_val - self.min_val

    def forward(self, x, out_y_only = True, quantize = True):
        total_bits = 0
        x_hat = torch.zeros_like(x)
        for j in range(x.shape[0]):
            for i in range(x.shape[1]):
                x_feat = x[j][i].detach().cpu().numpy()
                x_feat = (np.clip((x_feat-self.min_val)/self.dr, self.min_val, self.max_val)*255).astype('uint8')
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                bit_stream = cv2.imencode('.jpg', x_feat, encode_param)[1]
                n_bits = bit_stream.size*bit_stream.itemsize*8
                rec = cv2.imdecode(bit_stream, cv2.IMREAD_GRAYSCALE)
                x_hat[j,i,:,:] = torch.tensor(rec, device = x.device)
                total_bits += n_bits
        x_hat = (x_hat*self.dr/255+self.min_val)
                
        if out_y_only:
            if quantize:
                return x_hat
            else:
                return x
        else:
            if quantize:
                return {
                    "bits": total_bits,
                    "yhat": x_hat,
                }
            else:
                return {
                    "bits": 0,
                    "y": x,
                }

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
        
class Conv2(nn.Module):
    # Standard convolution no activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv2, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        
    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        return self.conv(x)
        
class ConvAct(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))
        
class TransposeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(TransposeConv, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, kernel_size=k, stride=s, output_padding=s - 1,
        padding=k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
        
class HyperUp(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, up=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(HyperUp, self).__init__()
        if up == True:
            self.conv1 = nn.ConvTranspose2d(c1, c1, kernel_size=5, stride=2, output_padding=1,
            padding=5 // 2, groups=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(c1, c1, 3, 1, autopad(3, 0))
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(c1, c2, 3, 1, 1)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(c2, c2, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x
        
class DeConv(nn.Module):
    # Deconvolution
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super(DeConv, self).__init__()
        self.conv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        return self.conv(x)
        
class SiLU(nn.Module):
    # Standard convolution
    def __init__(self):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SiLU, self).__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x)
        
class Conv_GDN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, inverse=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_GDN, self).__init__()
        self.inverse = inverse
        if inverse == False:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
            self.gdn = GDN(c2, inverse=inverse)
        elif inverse == True:
            self.conv = nn.ConvTranspose2d(c1, c2, kernel_size=k, stride=s, output_padding=s - 1,
                padding=k // 2, groups=g, bias=False)
            self.gdn = GDN(c2, inverse=inverse)

    def forward(self, x):
        if self.inverse == False:
            return self.gdn(self.conv(x))
        elif self.inverse == True:
            return self.conv(self.gdn(x))
        
class iGDN_Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(iGDN_Conv, self).__init__()
        self.igdn = GDN(c1, inverse = True)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        
    def forward(self, x):
        return self.conv(self.igdn(x))
        
class GDN_layer(nn.Module):
    # Standard convolution
    def __init__(self, c, inverse = 0):  # ch_in, ch_out, kernel, stride, padding, groups
        super(GDN_layer, self).__init__()
        self.gdn = GDN(c, inverse == 1)  ## inverse == 1: iGDN, inverse == 0 GDN

    def forward(self, x):
        return self.gdn(x)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        t.append(time_synchronized())

        # Post-process
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])
        t.append(time_synchronized())

        return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)
        self.t = ((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
              tuple(self.t))                      

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
