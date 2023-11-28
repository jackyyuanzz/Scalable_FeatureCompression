import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.utils import conv, deconv, update_registered_buffers

class LatentVisualizer_yolodirect3(nn.Module):

    def __init__(self, N, M, **kwargs):
        super().__init__()

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.LeakyReLU(),
            deconv(N, N),
            nn.LeakyReLU(),
            nn.Conv2d(N, 3, kernel_size = 3, padding = 1),
        )

        self.N = N
        self.M = M

    def forward(self, y_hat):
        x_hat = self.g_s(y_hat)

        return x_hat