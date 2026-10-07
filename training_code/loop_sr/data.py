"""Deterministic paired synthetic degradation; source-group train/val split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import random
import re
import signal

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, Sampler

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RESAMPLERS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
              "lanczos": Image.Resampling.LANCZOS}


def looks_preupscaled(name):
    # Conservative filename hints, not an image-quality classifier.
    return bool(re.search(r"\d\.\d{2}x_|ahq-\d|realesrgan|swinfir|waifu2x", name, re.IGNORECASE))


def ignore_worker_interrupt(worker_id):
    """Let the parent finish its current update and save after Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def image_to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def open_rgb(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def source_group(relative_path, video_sources=None):
    """Use extraction report when present; group all numbered sibling frames."""
    name = Path(relative_path).name
    if video_sources and name in video_sources:
        return "video:" + video_sources[name]
    stem = Path(relative_path).stem
    stem = re.sub(r"(?:_frame_\d+|_snapshot_).*$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return str(Path(relative_path).parent / stem).casefold()


def build_manifest(root, destination, seed=2026, val_fraction=0.02, min_size=192, verify=False,
                   include_preupscaled=False):
    root = Path(root).resolve()
    if not root.is_dir() or not 0 < val_fraction < 0.5:
        raise ValueError("Dataset directory and val_fraction in (0, 0.5) required")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Manifest already exists: {destination}; use a new filename to resplit")
    report = root / "frame_extraction_report.jsonl"
    video_sources = {}
    if report.exists():
        with report.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if row.get("output") and row.get("video"):
                        name = PureWindowsPath(row["output"]).name
                        video_sources[name] = str(row["video"]).casefold()
                except (ValueError, TypeError):
                    continue
    groups, skipped, extensions = defaultdict(list), [], Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        if not include_preupscaled and looks_preupscaled(path.name):
            skipped.append({"path": relative, "reason": "filename suggests an already upscaled HR target"})
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                if min(width, height) < min_size:
                    skipped.append({"path": relative, "reason": f"smaller than {min_size}px"})
                    continue
                if verify:
                    image.verify()
            group = source_group(relative, video_sources)
            # Compact group ID avoids embedding source video paths in the manifest.
            group = hashlib.sha256(group.encode("utf-8")).hexdigest()[:20]
            groups[group].append({"path": relative, "group": group, "width": width, "height": height})
            extensions[path.suffix.lower()] += 1
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            skipped.append({"path": relative, "reason": str(error)})
    if len(groups) < 2:
        raise ValueError("Need at least two independent source groups for train/validation")
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    target = max(1, round(sum(len(v) for v in groups.values()) * val_fraction))
    val_keys, count = set(), 0
    for key in keys[:-1]:  # Always leave at least one train group.
        if count >= target:
            break
        val_keys.add(key)
        count += len(groups[key])
    train, val = [], []
    for key in sorted(groups):
        (val if key in val_keys else train).extend(groups[key])
    manifest = dict(version=1, root=str(root), seed=seed, val_fraction=val_fraction,
                    min_size=min_size, verified=verify, include_preupscaled=include_preupscaled,
                    train=train, val=val, skipped=skipped,
                    counts={"train": len(train), "val": len(val), "groups": len(groups),
                            "val_groups": len(val_keys), "extensions": dict(extensions)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(destination)
    print(json.dumps(manifest["counts"]))
    print(f"Skipped: {len(skipped)}; manifest: {destination}")
    return manifest


def degrade(image, scale, cfg, rng, np_rng, clean=False):
    target = (image.width // scale, image.height // scale)
    if clean:
        return image.resize(target, Image.Resampling.BICUBIC)
    if rng.random() < cfg.get("clean_probability", 0.25):
        return image.resize(target, Image.Resampling.BICUBIC)
    if rng.random() < cfg.get("blur_probability", 0.25):
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*cfg.get("blur_sigma", [0.2, 1.0]))))
    if rng.random() < cfg.get("resize_probability", 0.3):
        ratio = rng.uniform(*cfg.get("resize_range", [0.75, 1.25]))
        image = image.resize(tuple(max(1, round(v * ratio)) for v in target),
                             RESAMPLERS[rng.choice(cfg.get("resamplers", list(RESAMPLERS)))])
    image = image.resize(target, RESAMPLERS[rng.choice(cfg.get("resamplers", list(RESAMPLERS)))])
    if rng.random() < cfg.get("noise_probability", 0.3):
        array = np.array(image, dtype=np.float32)
        sigma = rng.uniform(*cfg.get("noise_sigma", [0.2, 3.0]))
        shape = (*array.shape[:2], 1) if rng.random() < 0.5 else array.shape
        array += np_rng.normal(0, sigma, shape).astype(np.float32)
        image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
    if rng.random() < cfg.get("jpeg_probability", 0.3):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(*cfg.get("jpeg_quality", [75, 98])),
                   subsampling=rng.choice([0, 0, 2]))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
    return image


class AnimeDataset(Dataset):
    def __init__(self, manifest, split, scale, patch_size, degradation=None, seed=2026,
                 validation_mode="clean", limit=None, root=None):
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.root = Path(root or self.manifest["root"])
        self.items = self.manifest[split]
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError(f"Empty {split} split")
        train_groups = {row["group"] for row in self.manifest["train"]}
        val_groups = {row["group"] for row in self.manifest["val"]}
        if train_groups & val_groups:
            raise ValueError("Manifest leaks source groups between train and validation")
        if patch_size < scale * 2 or patch_size % scale:
            raise ValueError("patch_size must be divisible by scale and at least 2*scale")
        if validation_mode not in {"clean", "degraded"}:
            raise ValueError("validation_mode must be clean or degraded")
        self.scale, self.patch, self.seed = scale, patch_size, seed
        self.training, self.validation_mode = split == "train", validation_mode
        self.degradation = degradation or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        index, draw = key if isinstance(key, tuple) else (key, key)
        row = self.items[index]
        sample_seed = int.from_bytes(hashlib.blake2b(
            f"{self.seed}:{draw}:{row['path']}".encode(), digest_size=8).digest(), "little")
        rng, np_rng = random.Random(sample_seed), np.random.default_rng(sample_seed)
        path = self.root / row["path"]
        try:
            image = open_rgb(path)
        except (OSError, ValueError) as error:
            # Never silently substitute a different image, especially in validation.
            raise RuntimeError(f"Unreadable dataset image: {path}; rebuild with --verify") from error
        if min(image.size) < self.patch:
            raise ValueError(f"Image too small for patch {self.patch}: {path}; rebuild manifest with --min-size")
        if self.training:
            left, top = rng.randint(0, image.width - self.patch), rng.randint(0, image.height - self.patch)
        else:
            left, top = (image.width - self.patch) // 2, (image.height - self.patch) // 2
        image = image.crop((left, top, left + self.patch, top + self.patch))
        if self.training:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.ROTATE_90)
        lr = degrade(image, self.scale, self.degradation, rng, np_rng,
                     clean=not self.training and self.validation_mode == "clean")
        return image_to_tensor(lr), image_to_tensor(image)


class StepBatchSampler(Sampler):
    """Resumption does not depend on worker RNG, prefetch or epoch boundaries."""
    def __init__(self, size, batch_size, accumulation, max_steps, start_step, seed,
                 reference_batch_size=None):
        if min(size, batch_size, accumulation, max_steps) <= 0 or not 0 <= start_step <= max_steps:
            raise ValueError("Invalid sampler settings")
        self.size, self.batch = size, batch_size
        self.accumulation, self.max_steps, self.start_step, self.seed = accumulation, max_steps, start_step, seed
        self.reference_batch = reference_batch_size or batch_size
        if self.reference_batch < 1 or (batch_size * accumulation) % self.reference_batch:
            raise ValueError("Reference batch must divide effective batch")

    def __len__(self):
        return (self.max_steps - self.start_step) * self.accumulation

    def __iter__(self):
        effective = self.batch * self.accumulation
        reference_accumulation = effective // self.reference_batch
        for step in range(self.start_step, self.max_steps):
            samples = []
            for offset in range(reference_accumulation):
                micro = step * reference_accumulation + offset
                rng = random.Random(f"{self.seed}:{micro}")
                samples.extend((rng.randrange(self.size), micro * self.reference_batch + i)
                               for i in range(self.reference_batch))
            for offset in range(0, effective, self.batch):
                yield samples[offset:offset + self.batch]


def main():
    parser = argparse.ArgumentParser(description="Build a stable source-group split without changing original images")
    parser.add_argument("--root", default="data/hr")
    parser.add_argument("--output", default="data/manifests/anime_x2.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-size", type=int, default=192)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--include-preupscaled", action="store_true",
                        help="Include HR targets whose filenames suggest prior AI upscaling")
    args = parser.parse_args()
    build_manifest(args.root, args.output, args.seed, args.val_fraction, args.min_size, args.verify,
                   args.include_preupscaled)


if __name__ == "__main__":
    main()
