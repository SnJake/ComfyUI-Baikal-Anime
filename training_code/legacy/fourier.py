"""Retained SwinFIR v31 inference architecture; unchanged checkpoint keys."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm="ortho"):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        input_dtype = x.dtype
        if input_dtype not in (torch.float32, torch.float64):
            x = x.float()
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, ffted.shape[-2], ffted.shape[-1])
        conv_dtype = self.conv.weight.dtype
        if ffted.dtype != conv_dtype:
            ffted = ffted.to(dtype=conv_dtype)
        ffted = self.act(self.conv(ffted))
        ffted = ffted.view(batch, -1, 2, ffted.shape[-2], ffted.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
        if ffted.dtype not in (torch.float16, torch.float32, torch.float64):
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        out = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)
        return out

class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fourier = FourierUnit(embed_dim // 2)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1) if last_conv else None

    def forward(self, x):
        x_in = self.conv1(x)
        out = self.conv2(x_in + self.fourier(x_in))
        if self.last_conv is not None:
            out = self.last_conv(out)
        return out

class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x

class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super().__init__()
        self.spatial = ResB(embed_dim, red=red)
        self.spectral = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def forward(self, x):
        out = torch.cat([self.spatial(x), self.spectral(x)], dim=1)
        return self.fusion(out)
