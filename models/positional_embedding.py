import numpy as np
import torch
import torch.nn as nn

class FourierEncoding(nn.Module):
    def __init__(self, channels):
        
        super(FourierEncoding, self).__init__()
        self.channels = channels
        self.scales = 2**torch.arange(channels).reshape(1,-1)
        
    def forward(self, beta):
        batchsize = beta.shape[0]
        enc = beta.reshape(-1,1).repeat(1,self.channels)*self.scales.to(beta.device)*3.1416
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], 1)
        
        return enc
       