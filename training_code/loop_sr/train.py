"""Step-based single-GPU trainer with exact data-position resumption."""
import argparse
import copy
import json
import platform
import signal
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .data import AnimeDataset, StepBatchSampler, ignore_worker_interrupt
from .compilation import prepare_training_compile
from .evaluate import evaluate, validation_loader
from .losses import AnimeLoss
from .model import LoopedSR
from .progress import TrainingProgress
from .runtime import (EMA, amp_dtype, append_json, atomic_save, choose_device, file_digest,
    learning_rate, model_architecture, project_path, read_config, restore_rng, rng_state, seed_everything)


def assert_resume_compatible(saved, cfg, digest, payload):
    if model_architecture(saved["model"]) != model_architecture(cfg["model"]):
        raise ValueError("Resume changes model architecture; use --init for compatible weights")
    for section in ("loss", "optimizer"):
        if saved[section] != cfg[section]:
            raise ValueError(f"Resume changes {section}; use --init for a new fine-tuning run")
    for name in ("patch_size", "degradation"):
        if saved["data"][name] != cfg["data"][name]:
            raise ValueError(f"Resume changes data.{name}; use --init")
    previous_effective = saved["train"]["batch_size"] * saved["train"]["accumulation"]
    effective = cfg["train"]["batch_size"] * cfg["train"]["accumulation"]
    if effective != previous_effective:
        raise ValueError("Resume changes effective batch; keep batch_size * accumulation unchanged or use --init")
    for name in ("seed", "max_steps", "amp", "ema_decay"):
        if saved["train"][name] != cfg["train"][name]:
            raise ValueError(f"Resume changes train.{name}; use --init")
    if digest != payload["manifest_sha256"]:
        raise ValueError("Dataset manifest changed since checkpoint; use --init with a new output")


def run(cfg, device, resume=None, initialize=None, stop_after=None):
    tc, dc, oc = cfg["train"], cfg["data"], cfg["optimizer"]
    seed_everything(tc["seed"])
    if device.type == "cuda":
        torch.set_num_threads(tc.get("cpu_threads", 6))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    manifest = project_path(dc["manifest"])
    if not manifest.exists():
        raise FileNotFoundError(f"Prepare dataset first: python -m loop_sr.data --root {dc['root']} "
                                f"--output {dc['manifest']} --min-size {dc['patch_size']}")
    digest = file_digest(manifest)
    output = project_path(tc["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError(f"{output}/last.pt exists; use --resume or a new --output")
    dtype = amp_dtype(tc["amp"], device)
    dataset = AnimeDataset(manifest, "train", cfg["model"]["scale"], dc["patch_size"],
                           dc["degradation"], tc["seed"], root=dc.get("root"))
    model = LoopedSR(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=oc["lr"], betas=tuple(oc["betas"]),
                                 weight_decay=oc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    ema = EMA(model, tc["ema_decay"])
    criterion = AnimeLoss(**cfg["loss"]).to(device)
    step, best = 0, -1.0
    sampling_batch_size = tc["batch_size"]
    if resume or initialize:
        path = project_path(resume or initialize)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "loop_sr_v1":
            raise ValueError("Cannot load legacy SwinFIR weights into LoopedSR")
        if model_architecture(payload["config"]["model"]) != model_architecture(cfg["model"]):
            raise ValueError("Checkpoint architecture does not match config")
        if resume:
            assert_resume_compatible(payload["config"], cfg, digest, payload)
            model.load_state_dict(payload["model"], strict=True)
            ema.model.load_state_dict(payload["ema"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scaler.load_state_dict(payload["scaler"])
            step, best = payload["step"], payload["best_psnr"]
            sampling_batch_size = payload.get("sampling_batch_size", payload["config"]["train"]["batch_size"])
            restore_rng(payload["rng"])
        else:
            model.load_state_dict(payload["ema"], strict=True)
            ema.model.load_state_dict(payload["ema"], strict=True)
    if step >= tc["max_steps"]:
        print("Requested update limit already reached", flush=True)
        return
    sampler = StepBatchSampler(len(dataset), tc["batch_size"], tc["accumulation"],
                               tc["max_steps"], step, tc["seed"], sampling_batch_size)

    def compile_sample():
        samples = [dataset[index] for index in next(iter(sampler))]
        return tuple(torch.stack(items) for items in zip(*samples))

    training_forward, loss_forward, compile_info = prepare_training_compile(
        model, criterion, tc.get("compile"), device, dtype, compile_sample)
    loader_args = dict(dataset=dataset, batch_sampler=sampler, num_workers=tc["workers"],
                       pin_memory=device.type == "cuda")
    if tc["workers"]:
        loader_args.update(prefetch_factor=tc.get("prefetch_factor", 2), persistent_workers=True,
                           worker_init_fn=ignore_worker_interrupt)
    # Worker seeding must not consume the model's global RNG.
    loader_args["generator"] = torch.Generator().manual_seed(tc["seed"])
    loader = DataLoader(**loader_args)
    val_loaders = {mode: validation_loader(cfg, mode) for mode in cfg["validation"]["modes"]}
    if not val_loaders:
        raise ValueError("At least one validation mode required")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "device": str(device), "manifest_sha256": digest,
                   "parameters": sum(p.numel() for p in model.parameters()), "train_images": len(dataset)}
    environment["compile"] = compile_info
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment), flush=True)
    print(f"Training from update {step}; effective batch {tc['batch_size'] * tc['accumulation']}", flush=True)
    end = min(tc["max_steps"], step + stop_after) if stop_after else tc["max_steps"]
    progress = TrainingProgress(tc, step, end)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    updates_start, clock_start = step, time.perf_counter()
    micro_count, running = 0, {}
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            progress.write("Ctrl+C: finishing the current optimizer update, then saving")
        stop_requested = True

    def save(name):
        atomic_save({"format": "loop_sr_v1", "config": copy.deepcopy(cfg), "step": step,
            "best_psnr": best, "manifest_sha256": digest, "model": model.state_dict(),
            "ema": ema.model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(),
            "sampling_batch_size": sampling_batch_size}, output / name)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        if step >= end:
            print("Requested update limit already reached", flush=True)
            return
        for lr, hr in loader:
            rate = learning_rate(step, oc, tc["max_steps"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                predictions = training_forward(lr, return_all=True)
            loss, details = loss_forward(predictions, hr)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at update {step}")
            scaler.scale(loss / tc["accumulation"]).backward()
            for key, value in details.items():
                running[key] = running.get(key, 0.0) + value.item() / tc["accumulation"]
            micro_count += 1
            if micro_count % tc["accumulation"]:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            # Nonfinite gradients are rejected above; scaler cannot silently skip an update.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            ema.update(model)
            vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
            progress.update(running["total"], rate, vram)
            if stop_requested:
                optimizer.zero_grad(set_to_none=True)
                save("last.pt")
                progress.write(f"Stopped cleanly at completed update {step}")
                return
            if step % tc["log_every"] == 0 or step == updates_start + 1 or step == end:
                elapsed = time.perf_counter() - clock_start
                row = dict(step=step, lr=rate, grad_norm=norm.item(),
                           seconds_per_update=elapsed / max(1, step - updates_start), **running)
                if device.type == "cuda":
                    row["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                    row["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024 ** 3
                append_json(output / "train.jsonl", row)
                if progress.json_console or not progress.enabled:
                    progress.write(json.dumps(row))
            running = {}
            # Free training activations before checkpoint/validation.
            del predictions, loss, details, lr, hr
            if step % tc["save_every"] == 0:
                save("last.pt")  # Save before validation, preserving completed training.
            if step % tc["validate_every"] == 0 or step == end:
                metrics = evaluate(ema.model, val_loaders, cfg, device, dtype=dtype)
                row = {"step": step, "metrics": metrics}
                append_json(output / "validation.jsonl", row)
                if progress.json_console or not progress.enabled:
                    progress.write(json.dumps(row))
                else:
                    progress.validation(step, metrics, model.loops)
                score = sum(mode[f"loop_{model.loops}"]["psnr_rgb"] for mode in metrics.values()) / len(metrics)
                if score > best:
                    best = score
                    save("best.pt")
                save("last.pt")
            if step >= end or stop_requested:
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        print(f"Interrupted; saving completed update {step}", flush=True)
        save("last.pt")
        return
    finally:
        progress.close()
        signal.signal(signal.SIGINT, previous_handler)
    save("last.pt")
    print(f"Finished at update {step}; checkpoint {output / 'last.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train LoopedSR on a single GPU")
    parser.add_argument("--config", default="configs/loop_sr_x2_4070ti.yaml")
    parser.add_argument("--device", default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Full-state resume; training schedule/data must match")
    group.add_argument("--init", help="EMA weights only; fresh optimizer for fine-tuning")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Override train.compile.enabled; --no-compile restores eager execution")
    parser.add_argument("--stop-after", type=int, help="Stop cleanly after N additional optimizer updates")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.output:
        cfg["train"]["output"] = args.output
    if args.compile is not None:
        cfg["train"].setdefault("compile", {})["enabled"] = args.compile
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("workers must be non-negative")
        cfg["train"]["workers"] = args.workers
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be positive")
    run(cfg, choose_device(args.device), args.resume, args.init, args.stop_after)


if __name__ == "__main__":
    main()
