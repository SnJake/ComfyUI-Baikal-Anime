import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AnimeDataset
from .metrics import image_metrics
from .runtime import amp_dtype, choose_device, load_model, project_path, read_config


def validation_loader(cfg, mode, limit=None):
    dataset = AnimeDataset(project_path(cfg["data"]["manifest"]), "val", cfg["model"]["scale"],
        cfg["data"]["patch_size"], cfg["data"]["degradation"], cfg["train"]["seed"], mode,
        limit or cfg["validation"]["max_images"], root=cfg["data"].get("root"))
    return DataLoader(dataset, batch_size=cfg["validation"]["batch_size"], num_workers=0,
                      pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def evaluate(model, loaders, cfg, device, loops=None, dtype=None, per_image=None):
    selected = loops or cfg["validation"]["loops"]
    if not selected or any(not 1 <= n <= model.loops for n in selected):
        raise ValueError("Validation loops out of trained range")
    was_training = model.training
    model.eval()
    report = {}
    try:
        for mode, loader in loaders.items():
            totals, count = {}, 0
            with tqdm(loader, desc=f"Val/{mode}", unit="batch", leave=False,
                      dynamic_ncols=True, ascii=True,
                      disable=not cfg["train"].get("progress", True)) as batches:
                for lr, hr in batches:
                    lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                        exits = model(lr, return_all=True, loops=max(selected))
                    predictions = {f"loop_{n}": exits[n - 1] for n in selected}
                    predictions["bicubic"] = F.interpolate(lr, scale_factor=model.scale, mode="bicubic", align_corners=False)
                    sample_values = [{} for _ in range(hr.shape[0])] if per_image is not None else None
                    for name, prediction in predictions.items():
                        values = image_metrics(prediction, hr, cfg["validation"]["crop_border"],
                                               cfg["loss"]["flat_threshold"])
                        aggregate = totals.setdefault(name, {key: 0.0 for key in values})
                        for key, value in values.items():
                            aggregate[key] += value.double().sum().item()
                        if sample_values is not None:
                            cpu_values = {key: value.tolist() for key, value in values.items()}
                            for index, sample in enumerate(sample_values):
                                sample[name] = {key: value[index] for key, value in cpu_values.items()}
                    if sample_values is not None:
                        for index, sample in enumerate(sample_values):
                            item = loader.dataset.items[count + index]
                            per_image.append({"mode": mode, "path": item["path"], "group": item["group"],
                                              "metrics": sample})
                    count += hr.shape[0]
            report[mode] = {name: {key: value / count for key, value in values.items()}
                            for name, values in totals.items()}
            report[mode]["images"] = count
    finally:
        model.train(was_training)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate all supervised exits and bicubic on the same held-out data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional dataset/validation override; architecture is read from checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw", action="store_true", help="Use raw model instead of EMA")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--loops", type=int, nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--per-image", action="store_true", help="Save individual metrics for distribution analysis")
    args = parser.parse_args()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_num_threads(6)
    model, payload = load_model(args.checkpoint, device, args.raw)
    cfg = read_config(args.config) if args.config else payload["config"]
    if cfg["model"]["scale"] != model.scale:
        raise ValueError("Evaluation config scale differs from checkpoint")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    loaders = {mode: validation_loader(cfg, mode, args.limit) for mode in cfg["validation"]["modes"]}
    samples = [] if args.per_image else None
    report = evaluate(model, loaders, cfg, device, args.loops, amp_dtype(cfg["train"]["amp"], device), samples)
    report = {"checkpoint": str(args.checkpoint), "step": payload.get("step"),
              "weights": "raw" if args.raw else "ema", "metrics": report}
    print(json.dumps(report, indent=2, allow_nan=False))
    if samples is not None:
        report["per_image"] = samples
    if args.output:
        path = project_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
