"""Export EMA-only LoopSR safetensors with the architecture required for inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_code"))
from safetensors.torch import load_file, save_file
import torch

from loop_sr.model import LoopedSR
from loop_sr.runtime import project_path


def export(checkpoint, destination, model_version=None):
    checkpoint, destination = project_path(checkpoint), project_path(destination)
    if destination.suffix != ".safetensors" or destination.exists():
        raise ValueError("Choose a new .safetensors output file")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "loop_sr_v1":
        raise ValueError("Expected a LoopSR training checkpoint")
    config = dict(payload["config"]["model"], checkpoint_blocks=False)
    model = LoopedSR(**config)
    model.load_state_dict(payload["ema"], strict=True)
    weights = {name: value.detach().cpu().contiguous().clone() for name, value in payload["ema"].items()}
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("EMA contains nonfinite weights")
    metadata = {"architecture": "Baikal_LoopSR", "model_config": json.dumps(config),
                "training_steps": str(payload["step"]), "weights": "EMA",
                "format": "baikal_loopsr_v1", "license": "MIT"}
    if model_version is not None:
        metadata["model_version"] = str(model_version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(destination), metadata=metadata)
    restored = load_file(str(destination))
    assert weights.keys() == restored.keys() and all(torch.equal(value, restored[name]) for name, value in weights.items())
    print(json.dumps({"output": str(destination), "training_steps": payload["step"],
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "size_mib": destination.stat().st_size / 1024**2,
                      "ema_bit_exact": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.model_version)


if __name__ == "__main__":
    main()
