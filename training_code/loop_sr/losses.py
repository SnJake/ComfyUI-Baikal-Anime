"""Target-anchored losses for line art; no VGG, GAN or spectral magnitude loss."""
import torch
from torch import nn
from torch.nn import functional as F


def luminance(x):
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (x * weights).sum(1, keepdim=True)


def smooth(x):
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def gradients(x):
    """Signed gradients distinguish correct edges from ringing and texture."""
    return x[..., 1:] - x[..., :-1], x[..., 1:, :] - x[..., :-1, :]


def flat_mask(target, threshold=0.015):
    """Ground-truth-only mask, with a 5x5 exclusion around strong contours."""
    gray = luminance(target)
    dx, dy = gradients(gray)
    strength = F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))
    strength = F.max_pool2d(strength, 5, stride=1, padding=2)
    return torch.exp(-torch.square(strength / threshold)).detach()


class AnimeLoss(nn.Module):
    def __init__(self, pixel=1.0, edge=0.15, high_frequency=0.05, color=0.10,
                 flat=0.08, flat_threshold=0.015, charbonnier_eps=1e-3,
                 auxiliary_weight=1.0):
        super().__init__()
        if any(x < 0 for x in (pixel, edge, high_frequency, color, flat, auxiliary_weight)):
            raise ValueError("Loss weights must be non-negative")
        if pixel <= 0 or flat_threshold <= 0 or charbonnier_eps <= 0:
            raise ValueError("pixel, flat_threshold and charbonnier_eps must be positive")
        self.weights = dict(pixel=pixel, edge=edge, high_frequency=high_frequency, color=color, flat=flat)
        self.threshold, self.eps, self.auxiliary_weight = flat_threshold, charbonnier_eps, auxiliary_weight

    def components(self, prediction, target, mask):
        error = prediction - target
        pdx, pdy = gradients(luminance(prediction))
        tdx, tdy = gradients(luminance(target))
        hp = error - smooth(error)
        return {
            "pixel": (torch.sqrt(error.square() + self.eps ** 2) - self.eps).mean(),
            "edge": ((pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()) / 2,
            "high_frequency": hp.abs().mean(),
            "color": (smooth(smooth(prediction)) - smooth(smooth(target))).abs().mean(),
            # Error high-pass, not TV(prediction): correct target detail is never penalized.
            # Normalize each image separately: changing microbatch boundaries
            # must not reweight images according to their amount of flat area.
            "flat": ((hp.abs() * mask).flatten(1).sum(1) /
                     (mask.flatten(1).sum(1).clamp_min(1.0) * prediction.shape[1])).mean(),
        }

    def forward(self, predictions, target):
        if torch.is_tensor(predictions):
            predictions = [predictions]
        if not predictions:
            raise ValueError("At least one prediction required")
        # All loss arithmetic is FP32, even when the network runs in BF16/FP16.
        with torch.autocast(device_type=target.device.type, enabled=False):
            target = target.float()
            mask = flat_mask(target, self.threshold)
            terms = [self.components(p.float(), target, mask) for p in predictions]
            totals = [sum(self.weights[k] * v for k, v in t.items()) for t in terms]
            # Paper final+mean: (1/3, 1/3, 1/3, 1), not mean(all exits).
            total = totals[-1]
            if len(totals) > 1:
                total = total + self.auxiliary_weight * torch.stack(totals[:-1]).mean()
        details = {k: v.detach() for k, v in terms[-1].items()}
        details.update(total=total.detach(), final=totals[-1].detach())
        for i, value in enumerate(totals):
            details[f"exit_{i + 1}"] = value.detach()
        return total, details
