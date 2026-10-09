"""Standalone LoopSR safetensors inference; does not import or launch ComfyUI."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from training_code.loop_sr.model import LoopedSR
from baikal.tiling import upscale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--halo", type=int, default=96)
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError("Choose a new output file; input/existing results are protected")
    with safe_open(args.weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("architecture") != "Baikal_LoopSR":
        raise ValueError("Expected an exported Baikal LoopSR safetensors model")
    cfg = json.loads(metadata["model_config"])
    cfg["checkpoint_blocks"] = False
    model = LoopedSR(**cfg)
    model.load_state_dict(load_file(args.weights), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with Image.open(source) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2,0,1)[None].to(device)
    with torch.inference_mode():
        result = upscale(lambda x: model(x, loops=args.loops), tensor, model.scale, model.window,
                         args.tile, args.overlap, args.halo, lambda: None)
    array = result[0].permute(1,2,0).mul(255).round().byte().numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)
    print(f"Saved {destination}: {array.shape[1]}x{array.shape[0]}")


if __name__ == "__main__":
    main()
