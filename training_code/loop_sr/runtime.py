import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .model import LoopedSR

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def model_architecture(settings):
    """Activation checkpointing is a runtime choice, not a weight-layout change."""
    return {key: value for key, value in settings.items() if key != "checkpoint_blocks"}


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path):
    with project_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    for name in ("model", "data", "loss", "train", "optimizer", "validation"):
        if name not in cfg:
            raise ValueError(f"Missing config section {name}")
    train = cfg["train"]
    from .compilation import compile_settings

    compile_settings(train.get("compile"))
    for name in ("batch_size", "accumulation", "max_steps", "log_every", "save_every", "validate_every"):
        if int(train[name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    if train["amp"] not in {"bf16", "fp16", "off"}:
        raise ValueError("train.amp must be bf16, fp16 or off")
    scale = cfg["model"]["scale"]
    if scale not in (2, 3, 4):
        raise ValueError("model.scale must be 2, 3 or 4")
    if int(train.get("workers", 0)) < 0 or cfg["data"]["patch_size"] % scale:
        raise ValueError("Invalid workers or patch/scale")
    if train["grad_clip"] <= 0 or train.get("cpu_threads", 6) < 1:
        raise ValueError("grad_clip and cpu_threads must be positive")
    if cfg["optimizer"]["lr"] <= 0 or not 0 <= cfg["optimizer"]["min_lr"] <= cfg["optimizer"]["lr"]:
        raise ValueError("Require 0 <= min_lr <= lr, lr > 0")
    if not 0 <= cfg["optimizer"]["warmup_steps"] < train["max_steps"]:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    if not 0 <= train["ema_decay"] < 1 or cfg["validation"]["max_images"] < 1:
        raise ValueError("Invalid EMA or validation size")
    vc = cfg["validation"]
    if vc["batch_size"] < 1 or vc["crop_border"] < 0 or not vc["modes"]:
        raise ValueError("Invalid validation batch, border or modes")
    if not vc["loops"] or any(not 1 <= n <= cfg["model"]["loops"] for n in vc["loops"]):
        raise ValueError("Validation loops must be within the trained loop count")
    return cfg


def choose_device(requested="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype(amp, device):
    if amp == "off" or device.type != "cuda":
        return None
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; set train.amp to fp16 or off")
    return torch.bfloat16 if amp == "bf16" else torch.float16


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def append_json(path, payload):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for average, current in zip(self.model.parameters(), model.parameters()):
            average.lerp_(current.detach(), 1 - self.decay)
        for average, current in zip(self.model.buffers(), model.buffers()):
            average.copy_(current)


def learning_rate(step, cfg, total):
    warmup = cfg["warmup_steps"]
    if step < warmup:
        return cfg["lr"] * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))


def rng_state():
    # Only types accepted by torch.load(weights_only=True).
    py = random.getstate()
    state = np.random.get_state()
    return {"python": py, "numpy_name": state[0], "numpy_keys": torch.from_numpy(state[1].astype(np.int64)),
            "numpy_pos": state[2], "numpy_gauss": state[3], "numpy_cached": state[4],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_pos"], state["numpy_gauss"], state["numpy_cached"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_model(path, device, raw=False):
    payload = torch.load(project_path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "loop_sr_v1":
        raise ValueError("Expected a LoopSR checkpoint, not a legacy SwinFIR model")
    model = LoopedSR(**payload["config"]["model"])
    key = "model" if raw or "ema" not in payload else "ema"
    model.load_state_dict(payload[key], strict=True)
    return model.to(device).eval(), payload
