"""Retained SwinFIR v31 inference architecture; unchanged checkpoint keys."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .attention import SwinTransformerBlock
from .fourier import SFB

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if scale in (2, 3):
            m.append(nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1))
            m.append(nn.PixelShuffle(scale))
        elif scale == 4:
            for _ in range(2):
                m.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, h, w):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, h, w)

class ResidualSwinFourierBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path,
        patch_norm=True,
        resi_connection="SFB",
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        norm_layer = nn.LayerNorm if patch_norm else None
        self.patch_embed = PatchEmbed(dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(dim)

        connection = str(resi_connection).lower()
        if connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif connection == "hsfb":
            self.conv = SFB(dim, red=2)
        elif connection == "identity":
            self.conv = nn.Identity()
        else:
            self.conv = SFB(dim)

    def forward(self, x, h, w):
        residual = x
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.patch_unembed(x, h, w)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x + residual

class AnimeSwinFIR(nn.Module):
    def __init__(
        self,
        scale=2,
        in_channels=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="SFB",
    ):
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.img_range = float(img_range)
        self.upsampler = str(upsampler).lower()
        if in_channels == 3:
            self.register_buffer("mean", torch.tensor((0.3014, 0.3152, 0.3094)).view(1, 3, 1, 1), persistent=False)
        else:
            self.register_buffer("mean", torch.zeros(1, in_channels, 1, 1), persistent=False)

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        norm_layer = nn.LayerNorm if bool(patch_norm) else None
        self.patch_embed = PatchEmbed(embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_idx = 0
        for i in range(len(depths)):
            layer = ResidualSwinFourierBlock(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_idx : dpr_idx + depths[i]],
                patch_norm=bool(patch_norm),
                resi_connection=resi_connection,
            )
            dpr_idx += depths[i]
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.upsample = Upsample(scale, 64)
            self.conv_last = nn.Conv2d(64, in_channels, 3, 1, 1)
        else:
            self.conv_before_upsample = None
            self.upsample = None
            self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)

    def forward_features(self, x):
        b, _, h, w = x.shape
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, h, w)
        x = self.norm(x)
        return self.patch_unembed(x, h, w)

    def forward(self, x):
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        if self.upsampler == "pixelshuffle":
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        else:
            x = self.conv_last(x)

        return x / self.img_range + mean
