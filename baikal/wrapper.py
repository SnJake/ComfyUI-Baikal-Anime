from contextlib import nullcontext

import torch
from comfy import model_management, model_patcher


def autocast_context(device, mode):
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported; use auto/fp16/none")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    if dtype is None:
        raise ValueError(f"Unknown precision {mode}")
    return torch.autocast("cuda", dtype=dtype)


class BaikalModel:
    """ComfyUI custom-model and UPSCALE_MODEL compatible wrapper."""
    def __init__(self, model, kind, path, step=None):
        self.model, self.kind, self.path, self.step = model, kind, str(path), step
        self.scale = int(model.scale if kind == "loopsr" else 2)
        self.window = int(model.window if kind == "loopsr" else model.window_size)
        self.patcher = self._patcher(model_management.get_torch_device())

    def _patcher(self, device):
        return model_patcher.CoreModelPatcher(self.model, load_device=device,
                                             offload_device=model_management.unet_offload_device())

    def prepare(self, device, patch_pixels):
        if self.patcher.load_device != device:
            self.patcher = self._patcher(device)
        dim = self.model.stem.out_channels if self.kind == "loopsr" else self.model.conv_first.out_channels
        model_management.load_models_gpu([self.patcher], memory_required=patch_pixels * dim * 4 * 24,
                                         force_full_load=True)
        self.model.eval()

    def predict(self, image, loops=4):
        return self.model(image, loops=loops) if self.kind == "loopsr" else self.model(image)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, image):
        with torch.inference_mode(), autocast_context(image.device, "auto"):
            return self.predict(image)
