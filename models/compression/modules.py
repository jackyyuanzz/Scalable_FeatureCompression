import torch
import torch.nn as nn
import torch.nn.functional as F

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