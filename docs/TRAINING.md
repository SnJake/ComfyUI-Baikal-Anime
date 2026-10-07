# Training record

The released `Baikal_LoopSR_x2.safetensors` contains **EMA-only weights from update 24000**. Optimizer, scaler, RNG, file paths and dataset manifest are not embedded or distributed. Its safetensors metadata records architecture, model config, step, weight type and license.

Architecture: LR RGB input, bicubic residual, width 192, 6 heads (32/head), 8-pixel attention windows, 2 pre blocks, 4 shared middle blocks repeated 4 times, 2 post blocks. Unique parameters: 3,499,324. Final inference applies 20 blocks; training with all exits applies 26. XSA runs after SDPA and before head concatenation/output projection only in the shared stage. Normalization/projection arithmetic stays FP32 under autocast. Dense LR features and convolutional gated FFNs are restoration-specific choices.

Source: [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305), [official Looped-DiT code](https://github.com/OpenSenseNova/Looped-DiT). Shared computation, XSA and final+mean deep supervision are adapted; the diffusion sampler, text encoder, flow objective and original benchmark claims are not reproduced.

## Data and degradation

- 42,748 usable HR images, 41,893 training / 855 held-out, 30,756 source groups.
- Mostly anime and illustrations; a small rendered/3D fraction, exact percentage unknown.
- Training and validation use a source-group split.
- 128 validation images are fixed crops, representing 99 groups; the complete holdout is larger.
- Crops 192 HR / 96 LR; flips/rotations; clean bicubic pairs mixed with blur, resampling, mild Gaussian noise and JPEG on LR only. Targets are not automatically blurred/denoised.

## Optimization

AdamW: LR 2e-4, betas (0.9,0.99), weight decay 1e-4; warmup 2000, cosine schedule configured to 150000 updates, floor 2e-6; gradient clipping 1.0; EMA 0.999. Physical batch 4 and accumulation 4, effective batch 16 (the first short run used 8 x 2). BF16 on RTX 4070 Ti SUPER 16 GB; torch.compile enabled later. This artifact was exported after 24000 updates, not after the planned 150000 schedule.

| Loss | Weight |
|---|---:|
| Charbonnier RGB, epsilon 0.001 | 1.0 |
| Signed luminance gradients | 0.15 |
| High-pass prediction-minus-target error | 0.05 |
| Low-frequency RGB color | 0.10 |
| Per-image HR-flat-mask high-pass error | 0.08 |

The total uses final exit + mean of earlier exits. No perceptual/VGG, adversarial/GAN or FFT-magnitude term. Flat masking uses HR only with nearby contours excluded; normalization is per image so microbatch partitions do not reweight images. Correct HR detail is not penalized, but grain in HR can still be learned.

## Recorded validation, update 24000

| Mode | RGB PSNR | RGB SSIM | Flat HF error | Edge MAE |
|---|---:|---:|---:|---:|
| Clean synthetic LR | 42.479919 | 0.984089 | 0.001015807 | 0.006450925 |
| Degraded synthetic LR | 40.162320 | 0.976099 | 0.001075730 | 0.007436493 |

EMA loop 4, 128 fixed center crops, 2 HR-pixel border. PSNR averages per-image scores; SSIM uses a Gaussian window. These scores do not apply to the visual examples, which have no true HR reference. They do not establish superiority over other upscalers.
