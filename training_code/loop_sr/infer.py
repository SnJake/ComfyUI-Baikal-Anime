"""Whole-image or grid-aligned tiles with context halos and cosine blending."""
import argparse
import math
from itertools import product
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import functional as F

from .data import EXTENSIONS, image_to_tensor, open_rgb
from .runtime import amp_dtype, choose_device, load_model


def starts(length, tile, stride):
    # Keep EVERY start on the global attention grid, including the final tile.
    return [0] if length <= tile else list(range(0, length, stride))


def blend_axis(length, overlap, before, after, device):
    weight = torch.ones(length, device=device)
    overlap = min(overlap, length)
    if overlap:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, overlap + 2, device=device)[1:-1])
        if before:
            weight[:overlap] *= ramp
        if after:
            weight[-overlap:] *= ramp.flip(0)
    return weight


@torch.inference_mode()
def upscale(model, image, device, tile=192, overlap=32, halo=48, loops=None, dtype=None, progress=False):
    """Assemble output on CPU; only one tile lives on GPU at a time.

    Tiles are approximate: finite halo cannot guarantee whole-image equivalence
    for a deep recurrent network. Use tile=0 for a full forward pass.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Inference requires one BCHW image")
    if tile < 0 or halo < 0 or overlap < 0:
        raise ValueError("tile, overlap and halo must be non-negative")
    window, scale = model.window, model.scale
    model.eval()

    def forward(x):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            return model(x.to(device), loops=loops).float().cpu()

    if tile == 0:
        return forward(image).clamp(0, 1)
    if tile % window or overlap % window or halo % window or overlap >= tile:
        raise ValueError(f"tile/overlap/halo must be multiples of window={window}, with overlap < tile")
    image = image.cpu()
    h, w = image.shape[-2:]
    ph, pw = (-h) % window, (-w) % window
    mode = "reflect" if h > ph and w > pw else "replicate"
    image = F.pad(image, (0, pw, 0, ph), mode=mode) if ph or pw else image
    height, width = image.shape[-2:]
    canvas = torch.zeros((1, 3, height * scale, width * scale))
    weights = torch.zeros((1, 1, height * scale, width * scale))
    ys, xs = starts(height, tile, tile - overlap), starts(width, tile, tile - overlap)
    for y, x in tqdm(product(ys, xs), total=len(ys) * len(xs), desc="Upscale",
                     unit="tile", dynamic_ncols=True, ascii=True, disable=not progress):
        y1, x1 = min(height, y + tile), min(width, x + tile)
        top, left = max(0, y - halo), max(0, x - halo)
        bottom, right = min(height, y1 + halo), min(width, x1 + halo)
        patch = forward(image[..., top:bottom, left:right])
        patch = patch[..., (y - top) * scale:(y1 - top) * scale,
                      (x - left) * scale:(x1 - left) * scale]
        wy = blend_axis((y1 - y) * scale, overlap * scale, y > 0, y1 < height, "cpu")
        wx = blend_axis((x1 - x) * scale, overlap * scale, x > 0, x1 < width, "cpu")
        weight = (wy[:, None] * wx[None, :])[None, None]
        canvas[..., y * scale:y1 * scale, x * scale:x1 * scale] += patch * weight
        weights[..., y * scale:y1 * scale, x * scale:x1 * scale] += weight
    return (canvas / weights.clamp_min(1e-8))[..., :h * scale, :w * scale].clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Upscale an image or directory using EMA weights")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile", type=int, default=192, help="LR pixels; 0 means full image")
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--halo", type=int, default=48)
    parser.add_argument("--loops", type=int)
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_num_threads(6)
    model, _ = load_model(args.checkpoint, device, args.raw)
    source, destination = Path(args.input), Path(args.output)
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images: {source}")
    if source.is_dir() and destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output directory must be outside input directory")
    for path in paths:
        result = destination if source.is_file() else destination / path.relative_to(source).with_suffix(".png")
        if result.suffix.lower() != ".png":
            raise ValueError("Output must use lossless .png")
        if result.resolve() == path.resolve():
            raise ValueError("Output cannot replace the source image")
        if result.exists() and not args.overwrite:
            raise FileExistsError(f"{result}; use --overwrite to replace")
        image = image_to_tensor(open_rgb(path)).unsqueeze(0)
        prediction = upscale(model, image, device, args.tile, args.overlap, args.halo, args.loops,
                             amp_dtype(args.amp, device), progress=True)
        array = prediction.squeeze(0).permute(1, 2, 0).mul(255).round().byte().numpy()
        result.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(result)
        print(f"{path.name} -> {result} ({array.shape[1]}x{array.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
