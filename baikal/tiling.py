"""Window-aligned halo tiles with cosine blending and CPU assembly."""
import math
import torch
from torch.nn import functional as F


def starts(length, tile, stride):
    return [0] if length <= tile else list(range(0, length, stride))


def weights(length, overlap, before, after):
    result = torch.ones(length)
    overlap = min(overlap, length)
    if overlap:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, overlap + 2)[1:-1])
        if before:
            result[:overlap] *= ramp
        if after:
            result[-overlap:] *= ramp.flip(0)
    return result


@torch.inference_mode()
def upscale(predict, image, scale, window, tile, overlap, halo, on_tile):
    h, w = image.shape[-2:]
    ph, pw = (-h) % window, (-w) % window
    mode = "reflect" if h > ph and w > pw else "replicate"
    image = F.pad(image, (0, pw, 0, ph), mode=mode) if ph or pw else image
    height, width = image.shape[-2:]
    if not tile:
        result = predict(image).float().cpu()
        on_tile()
        return result[..., :h * scale, :w * scale].clamp(0, 1)
    tile = max(window * 2, tile // window * window)
    overlap = min(overlap // window * window, tile - window)
    halo = math.ceil(halo / window) * window
    canvas = torch.zeros(1, 3, height * scale, width * scale)
    blend = torch.zeros(1, 1, height * scale, width * scale)
    for y in starts(height, tile, tile - overlap):
        for x in starts(width, tile, tile - overlap):
            y1, x1 = min(height, y + tile), min(width, x + tile)
            top, left = max(0, y - halo), max(0, x - halo)
            bottom, right = min(height, y1 + halo), min(width, x1 + halo)
            result = predict(image[..., top:bottom, left:right]).float().cpu()
            result = result[..., (y - top) * scale:(y1 - top) * scale,
                            (x - left) * scale:(x1 - left) * scale]
            wy = weights((y1 - y) * scale, overlap * scale, y > 0, y1 < height)
            wx = weights((x1 - x) * scale, overlap * scale, x > 0, x1 < width)
            weight = (wy[:, None] * wx[None, :])[None, None]
            canvas[..., y * scale:y1 * scale, x * scale:x1 * scale] += result * weight
            blend[..., y * scale:y1 * scale, x * scale:x1 * scale] += weight
            on_tile()
    canvas.div_(blend.clamp_min(1e-8)).clamp_(0, 1)
    return canvas[..., :h * scale, :w * scale]
