import json
from pathlib import Path

import folder_paths
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file

from ..training_code.loop_sr.model import LoopedSR
from ..training_code.legacy.swinfir import AnimeSwinFIR
from .catalog import MODELS, MODEL_REPO
from .wrapper import BaikalModel

_CACHE = {}


def model_directory():
    path = Path(folder_paths.models_dir) / "anime_upscale"
    path.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("anime_upscale", str(path))
    return path


def resolve_weights(filename):
    if filename not in MODELS:
        raise ValueError("Supported models: " + ", ".join(MODELS))
    root = model_directory()
    path = folder_paths.get_full_path("anime_upscale", filename)
    if path:
        return Path(path)
    try:
        result = hf_hub_download(MODEL_REPO, filename=filename, revision="main", local_dir=str(root))
    except EntryNotFoundError:
        # After HF dev is merged, main resolves without a code/config change.
        if MODELS[filename]["kind"] != "loopsr":
            raise
        print(f"[Baikal] {filename} absent on main; downloading the dev preview.")
        result = hf_hub_download(MODEL_REPO, filename=filename, revision="dev", local_dir=str(root))
    return Path(result)


def load_model(filename, force_reload=False):
    path = resolve_weights(filename)
    stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if not force_reload and _CACHE.get(filename, (None,))[0] == stamp:
        return _CACHE[filename][1]
    spec = MODELS[filename]
    cfg = json.loads(spec["config"].read_text(encoding="utf-8"))
    step = None
    if spec["kind"] == "loopsr":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("architecture") != "Baikal_LoopSR":
            raise ValueError("Expected an exported Baikal LoopSR checkpoint")
        cfg = json.loads(metadata.get("model_config", json.dumps(cfg)))
        cfg["checkpoint_blocks"] = False
        model = LoopedSR(**cfg)
        step = int(metadata["training_steps"]) if "training_steps" in metadata else None
    else:
        parameters = dict(cfg["model"])
        parameters.pop("type", None)
        model = AnimeSwinFIR(scale=cfg["scale"], **parameters)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    model.eval().requires_grad_(False)
    wrapper = BaikalModel(model, spec["kind"], path, step)
    _CACHE[filename] = (stamp, wrapper)
    return wrapper
