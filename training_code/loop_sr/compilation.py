"""Training-only torch.compile; checkpoints always use the original modules."""
import os
import time

import torch

from .runtime import project_path, restore_rng, rng_state


def compile_settings(settings=None):
    settings = settings or {}
    if not isinstance(settings, dict):
        raise ValueError("train.compile must be a mapping")
    values = dict(enabled=False, backend="inductor", mode="default", fullgraph=False,
                  dynamic=False, compile_loss=False, fallback_on_error=True,
                  cache_dir=".cache/torch_compile")
    unknown = settings.keys() - values.keys()
    if unknown:
        raise ValueError(f"Unknown train.compile settings: {sorted(unknown)}")
    values.update(settings)
    if values["backend"] not in {"inductor", "aot_eager", "eager"}:
        raise ValueError("compile backend must be inductor, aot_eager or eager")
    if values["mode"] not in {"default", "max-autotune-no-cudagraphs"}:
        raise ValueError("Use default or max-autotune-no-cudagraphs to avoid CUDA graph memory overhead")
    for name in ("enabled", "fullgraph", "dynamic", "compile_loss", "fallback_on_error"):
        if not isinstance(values[name], bool):
            raise ValueError(f"train.compile.{name} must be boolean")
    return values


def _prime_masks(model, lr):
    """Avoid mutating Python mask caches inside the traced training forward."""
    h, w = lr.shape[-2:]
    h, w = h + (-h) % model.window, w + (-w) % model.window
    for stage in (model.pre, model.body, model.post):
        for block in stage:
            if block.shift and min(h, w) > block.window:
                block._mask(h, w, block.shift, lr.device)


def prepare_training_compile(model, criterion, settings, device, dtype, sample_factory):
    """Warm forward AND backward without updating weights or consuming data/RNG.

    Return separate callables; EMA, validation, optimizer and state_dict keep
    referencing the original modules. Lazy compilation errors are caught during
    warmup, never halfway through a real accumulated optimizer update.
    """
    settings = compile_settings(settings)
    info = {"requested": settings["enabled"], "active": False, "settings": settings}
    if not settings["enabled"]:
        return model, criterion, info
    if device.type != "cuda" and settings["backend"] == "inductor":
        info["reason"] = "Inductor training is enabled only on CUDA"
        print("Compile: CPU run uses eager execution", flush=True)
        return model, criterion, info
    state = rng_state()
    started = time.perf_counter()
    lr = hr = predictions = loss = None
    try:
        if settings["backend"] == "inductor":
            import triton

            info["triton"] = str(triton.__version__)
            cache = project_path(settings["cache_dir"]).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
            os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
            info["cache_dirs"] = {"inductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
                                  "triton": os.environ["TRITON_CACHE_DIR"]}
            options = dict(torch._inductor.list_mode_options(settings["mode"]))
            options["triton.cudagraphs"] = False
            compile_args = {"options": options}
        else:
            compile_args = {}
        compile_args.update(backend=settings["backend"], fullgraph=settings["fullgraph"],
                            dynamic=settings["dynamic"])
        print(f"Compile: {settings['backend']} / {settings['mode']}; warming forward/backward "
              "without optimizer updates. The first compilation can take several minutes.", flush=True)
        lr, hr = sample_factory()
        lr, hr = lr.to(device), hr.to(device)
        info["input_shape"] = list(lr.shape)
        _prime_masks(model, lr)
        forward = torch.compile(model, **compile_args)
        loss_forward = torch.compile(criterion, **compile_args) if settings["compile_loss"] else criterion
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            predictions = forward(lr, return_all=True)
        loss, _ = loss_forward(predictions, hr)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite loss during compile warmup")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        info.update(active=True, warmup_seconds=time.perf_counter() - started)
        print(f"Compile: ready after {info['warmup_seconds']:.1f}s; weights and optimizer unchanged", flush=True)
        return forward, loss_forward, info
    except Exception as error:
        info.update(error=f"{type(error).__name__}: {error}", warmup_seconds=time.perf_counter() - started)
        if not settings["fallback_on_error"]:
            raise
        print(f"Compile failed; using eager execution. {info['error'][:900]}", flush=True)
        return model, criterion, info
    finally:
        del lr, hr, predictions, loss
        model.zero_grad(set_to_none=True)
        restore_rng(state)
