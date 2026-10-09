# Training record

Released files contain EMA-only weights. The architecture and training step are recorded in safetensors metadata.

| Model | EMA update | Parameters | Width / heads | Pre / shared / post blocks |
|---|---:|---:|---|---|
| LoopSR v2 | 30000 | 5,234,987 | 224 / 7 | 3 / 4 / 2 |
| LoopSR v1 | 24000 | 3,499,324 | 192 / 6 | 2 / 4 / 2 |

Both models use LR RGB input, a bicubic residual, 8-pixel attention windows and four passes through one shared middle stage. XSA is applied only in the shared stage, after SDPA and before head concatenation. Inference can exit at loops 1–4. V2 was trained from scratch.

Idea: [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305), [official code](https://github.com/OpenSenseNova/Looped-DiT). Shared computation, XSA and deep supervision are adapted for direct RGB restoration.

## Data and optimization

42,748 usable HR images: 41,893 training / 855 held-out. Mostly anime and illustrations, with a small 3D/rendered fraction. Source groups are kept separate between training and validation. HR crops are 192 pixels; LR crops are 96. Training combines clean bicubic pairs and mild LR blur, resampling, noise and JPEG degradation.

AdamW: LR 2e-4, minimum 2e-6, warmup 2000 updates, betas (0.9, 0.99), weight decay 1e-4, gradient clipping 1.0. Physical batch 4, accumulation 4, effective batch 16. EMA 0.999, BF16, torch.compile, RTX 4070 Ti SUPER 16 GB. The cosine schedule is configured for 40000 updates in v2 and 150000 in v1; exported weights come from the updates listed above.

| Loss | v1 | v2 |
|---|---:|---:|
| Charbonnier RGB, epsilon 0.001 | 1.0 | 1.0 |
| Signed luminance gradients | 0.15, scale 1× | 0.20, scales 1× and ½× |
| High-pass prediction-minus-target error | 0.05 | 0.06 |
| Low-frequency RGB color | 0.10 | 0.10 |
| Per-image flat-region high-pass error | 0.08 | 0.08 |

V2's half-resolution edge term uses a 2×2 averaging filter before downsampling. Gradients are normalized to HR pixel spacing, then the two scales are averaged. Deep supervision is final exit + mean of earlier exits. Neither version uses VGG/perceptual, GAN or FFT-magnitude loss.

## Recorded validation

EMA loop 4 on the same 128 fixed held-out crops per mode, with a 2-pixel HR border. PSNR averages per-image scores. These synthetic-pair measurements are separate from the visual examples.

| Model | Mode | RGB PSNR | RGB SSIM | Edge MAE | Flat HF error |
|---|---|---:|---:|---:|---:|
| v1 | clean | 42.479919 | 0.984089 | 0.006450925 | 0.001015807 |
| v1 | degraded | 40.162320 | 0.976099 | 0.007436493 | 0.001075730 |
| v2 | clean | 42.789825 | 0.984565 | 0.006278376 | 0.001009161 |
| v2 | degraded | 40.429077 | 0.977069 | 0.007247928 | 0.001069068 |

## Training reference

Install `training_code/requirements.txt`, configure your dataset root and manifest, then from `training_code`:

```bash
python -m loop_sr.train --config configs/loop_sr_x2_v2.yaml
```

The v1 config remains available as `configs/loop_sr_x2.yaml`. For EMA export, run `tools/export_model.py` from the repository root with `--checkpoint`, `--output` and optional `--model-version`.
