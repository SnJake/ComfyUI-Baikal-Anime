"""RGB metrics on [0,1]; average per-image PSNR (not PSNR of mean MSE)."""
import torch
from torch.nn import functional as F

from .losses import flat_mask, gradients, luminance, smooth


def psnr(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def ssim(prediction, target):
    size = min(11, *target.shape[-2:])
    size -= 1 - size % 2
    position = torch.arange(size, device=target.device, dtype=torch.float32) - size // 2
    gaussian = torch.exp(-position.square() / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, size, size)

    def average(x):
        return F.conv2d(x, kernel, groups=3)

    mu_p, mu_t = average(prediction), average(target)
    var_p = (average(prediction.square()) - mu_p.square()).clamp_min(0)
    var_t = (average(target.square()) - mu_t.square()).clamp_min(0)
    covariance = average(prediction * target) - mu_p * mu_t
    score = ((2 * mu_p * mu_t + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
        (mu_p.square() + mu_t.square() + 0.01 ** 2) * (var_p + var_t + 0.03 ** 2))
    return score.flatten(1).mean(1)


def image_metrics(prediction, target, border=2, flat_threshold=0.015):
    prediction, target = prediction.float().clamp(0, 1), target.float()
    if border:
        if min(target.shape[-2:]) <= 2 * border + 1:
            raise ValueError("Metric crop border too large")
        prediction = prediction[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    mask = flat_mask(target, flat_threshold)
    error = prediction - target
    hp = error - smooth(error)
    pdx, pdy = gradients(luminance(prediction))
    tdx, tdy = gradients(luminance(target))
    return {
        "psnr_rgb": psnr(prediction, target),
        "ssim_rgb": ssim(prediction, target),
        "flat_hf_error": (hp.abs() * mask).flatten(1).sum(1) / (3 * mask.flatten(1).sum(1).clamp_min(1)),
        "edge_mae": ((pdx - tdx).abs().flatten(1).mean(1) + (pdy - tdy).abs().flatten(1).mean(1)) / 2,
    }
