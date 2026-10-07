"""Restoration adaptation of Looped-DiT sections 2.1/2.2 (not a diffusion model).

A = stem + pre blocks; B = shared window Transformer blocks; C = shared post
blocks + reconstruction head. Only B uses self-modulating attention.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def exclusive_self_attention(out, value):
    """Paper eq. (5): orthogonal projection, before head concatenation.

    Keep normalization AND subtraction in FP32 under autocast. This is not
    diagonal attention masking and does not change the softmax weights.
    """
    direction = F.normalize(value.float(), dim=-1, eps=1e-12)
    work = out.float()
    return (work - (work * direction).sum(-1, keepdim=True) * direction).to(out.dtype)


def partition(x, window):
    b, h, w, c = x.shape
    return x.reshape(b, h // window, window, w // window, window, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def unpartition(x, window, b, h, w):
    return x.reshape(b, h // window, w // window, window, window, -1).permute(
        0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window, modulation="none"):
        super().__init__()
        if dim % heads or dim // heads < 2:
            raise ValueError("dim must be divisible by heads, with head dimension >= 2")
        if modulation not in {"none", "xsa", "gated"}:
            raise ValueError("attention must be none, xsa or gated")
        self.heads, self.window, self.modulation = heads, window, modulation
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, heads) if modulation == "gated" else None
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        delta = coords.flatten(1)[:, :, None] - coords.flatten(1)[:, None, :]
        delta = delta.permute(1, 2, 0) + window - 1
        index = delta[..., 0] * (2 * window - 1) + delta[..., 1]
        self.register_buffer("relative_index", index.long(), persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x, mask=None):
        n, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(n, length, 3, self.heads, dim // self.heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(
            length, length, self.heads).permute(2, 0, 1).to(q.dtype)
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0))
        else:
            # Flatten batch/windows: CUDA SDPA kernels expect 4D Q/K/V.
            nw = mask.shape[0]
            additive = (bias[None] + mask.to(q.dtype)[:, None]).repeat(n // nw, 1, 1, 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive)
        if self.modulation == "xsa":
            out = exclusive_self_attention(out, v)
        elif self.modulation == "gated":
            out = out * self.gate(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        return self.proj(out.transpose(1, 2).reshape(n, length, dim))


class RestorationBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, ffn_ratio, modulation):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window, modulation)
        hidden = int(dim * ffn_ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hidden * 2), nn.Linear(hidden, dim)
        self.local = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.attn_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((dim,), 0.1))
        self._masks = OrderedDict()

    def _mask(self, h, w, shift, device):
        key = (h, w, shift, str(device))
        if key not in self._masks:
            labels = torch.zeros((1, h, w, 1), device=device)
            ws = self.window
            regions = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            for i, hs in enumerate(regions):
                for j, vs in enumerate(regions):
                    labels[:, hs, vs] = 3 * i + j
            labels = partition(labels, ws).squeeze(-1)
            blocked = labels[:, :, None] != labels[:, None, :]
            self._masks[key] = torch.zeros_like(blocked, dtype=torch.float32).masked_fill(blocked, -float("inf"))
            if len(self._masks) > 4:
                self._masks.popitem(last=False)
        return self._masks[key]

    def forward(self, x):
        b, c, h, w = x.shape
        stream = x.permute(0, 2, 3, 1)
        y = self.norm1(stream)
        shift = self.shift if min(h, w) > self.window else 0
        mask = self._mask(h, w, shift, x.device) if shift else None
        if shift:
            y = torch.roll(y, (-shift, -shift), (1, 2))
        y = unpartition(self.attn(partition(y, self.window), mask), self.window, b, h, w)
        if shift:
            y = torch.roll(y, (shift, shift), (1, 2))
        stream = stream + y * self.attn_scale
        y = self.fc1(self.norm2(stream)).permute(0, 3, 1, 2)
        a, gate = self.local(y).permute(0, 2, 3, 1).chunk(2, dim=-1)
        stream = stream + self.fc2(F.gelu(a) * gate) * self.ffn_scale
        return stream.permute(0, 3, 1, 2).contiguous()


class LoopedSR(nn.Module):
    def __init__(self, scale=2, dim=192, heads=6, window=8, pre_blocks=2,
                 loop_blocks=4, post_blocks=2, loops=4, ffn_ratio=2.0,
                 attention="xsa", checkpoint_blocks=True):
        super().__init__()
        if scale not in (2, 3, 4) or min(dim, heads, window, loop_blocks, loops) <= 0:
            raise ValueError("Invalid model dimensions/scale/depth")
        if pre_blocks < 0 or post_blocks < 0 or ffn_ratio <= 0:
            raise ValueError("Invalid block counts or FFN ratio")
        self.scale, self.window, self.loops = scale, window, loops
        self.checkpoint_blocks = checkpoint_blocks
        self.stem = nn.Conv2d(3, dim, 3, padding=1)

        def blocks(count, modulation):
            return nn.ModuleList([RestorationBlock(dim, heads, window,
                0 if i % 2 == 0 else window // 2, ffn_ratio, modulation) for i in range(count)])

        self.pre = blocks(pre_blocks, "none")
        self.body = blocks(loop_blocks, attention)  # One B, reused; never cloned per loop.
        self.post = blocks(post_blocks, "none")
        self.refine = nn.Conv2d(dim, dim, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(dim, 64, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(64, 3 * scale * scale, 3, padding=1), nn.PixelShuffle(scale))
        nn.init.normal_(self.head[2].weight, std=1e-3)
        nn.init.zeros_(self.head[2].bias)

    def _stage(self, state, blocks):
        for block in blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                state = checkpoint(block, state, use_reentrant=False, preserve_rng_state=False)
            else:
                state = block(state)
        return state

    def forward(self, lr, return_all=False, loops=None):
        if lr.ndim != 4 or lr.shape[1] != 3 or min(lr.shape[-2:]) < 1:
            raise ValueError("Expected nonempty BCHW RGB input")
        count = self.loops if loops is None else int(loops)
        if not 1 <= count <= self.loops:
            raise ValueError(f"loops must be in [1, {self.loops}]; extrapolation is not validated")
        h, w = lr.shape[-2:]
        ph, pw = (-h) % self.window, (-w) % self.window
        # Replicate works even for 1-pixel inputs; reflection needs enough pixels.
        mode = "reflect" if h > ph and w > pw else "replicate"
        padded = F.pad(lr, (0, pw, 0, ph), mode=mode) if ph or pw else lr
        baseline = F.interpolate(padded, scale_factor=self.scale, mode="bicubic", align_corners=False)
        shallow = self.stem(padded - 0.5)
        state = self._stage(shallow, self.pre)
        outputs = []
        for i in range(count):
            state = self._stage(state, self.body)
            if return_all or i == count - 1:
                decoded = self._stage(state, self.post)
                prediction = baseline + self.head(self.refine(decoded) + shallow)
                outputs.append(prediction[:, :, :h * self.scale, :w * self.scale])
        return outputs if return_all else outputs[-1]
