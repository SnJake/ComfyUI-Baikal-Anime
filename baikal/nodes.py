import math
import torch
from comfy import model_management, utils
from .catalog import MODELS
from .devices import resolve_device
from .loading import load_model, model_directory
from .tiling import starts, upscale
from .wrapper import autocast_context

CATEGORY = "Baikal/Upscale"


class BaikalModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_directory()
        return {"required": {"weights_name": (list(MODELS),), "force_reload": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("ANIME_UPSCALE_MODEL", "UPSCALE_MODEL")
    RETURN_NAMES = ("baikal_model", "upscale_model")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, weights_name, force_reload=False):
        model = load_model(weights_name, force_reload)
        return model, model


class BaikalUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale_model_custom": ("ANIME_UPSCALE_MODEL",), "image": ("IMAGE",),
            "tile": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8}),
            "overlap": ("INT", {"default": 32, "min": 0, "max": 1024, "step": 8}),
            "amp": (["auto", "bf16", "fp16", "none"], {"default": "auto"}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }, "optional": {
            "loops": ("INT", {"default": 4, "min": 1, "max": 4,
                               "tooltip": "LoopSR depth; ignored by SwinFIR."}),
            "halo": ("INT", {"default": 96, "min": 0, "max": 512, "step": 8,
                              "tooltip": "LR context around tiles; increase to reduce boundary differences."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    @torch.inference_mode()
    def upscale(self, upscale_model_custom, image, tile=256, overlap=32, amp="auto", device="auto", loops=4, halo=96):
        if image.ndim != 4 or image.shape[-1] != 3 or min(image.shape[:3]) < 1:
            raise ValueError("Expected a nonempty ComfyUI RGB IMAGE [B,H,W,3]")
        if min(tile, overlap, halo) < 0 or (tile and overlap >= tile):
            raise ValueError("Require tile/overlap/halo >= 0 and overlap < tile")
        wrapper = upscale_model_custom
        if wrapper.kind == "loopsr" and not 1 <= loops <= wrapper.model.loops:
            raise ValueError("Loop depth outside the trained range")
        target = resolve_device(model_management.get_torch_device() if device == "auto" else device)
        results, current_tile = [], int(tile)
        for item in image:
            while True:
                try:
                    h, w = item.shape[:2]
                    patch = (current_tile + 2 * halo) ** 2 if current_tile else h * w
                    wrapper.prepare(target, patch)
                    window = wrapper.window
                    hp, wp = math.ceil(h / window) * window, math.ceil(w / window) * window
                    actual_tile = max(window * 2, current_tile // window * window) if current_tile else 0
                    actual_overlap = min(overlap // window * window, actual_tile - window) if actual_tile else 0
                    steps = (len(starts(hp, actual_tile, actual_tile - actual_overlap)) *
                             len(starts(wp, actual_tile, actual_tile - actual_overlap))) if actual_tile else 1
                    progress = utils.ProgressBar(steps)

                    def on_tile():
                        model_management.throw_exception_if_processing_interrupted()
                        progress.update(1)

                    tensor = item.permute(2, 0, 1).contiguous()[None].float().to(target)
                    with autocast_context(target, amp):
                        result = upscale(lambda x: wrapper.predict(x, loops), tensor, wrapper.scale,
                                         window, actual_tile, overlap, halo, on_tile)
                    results.append(result[0].permute(1, 2, 0).contiguous())
                    break
                except Exception as error:
                    model_management.raise_non_oom(error)
                    current_tile = current_tile // 2 if current_tile else 256
                    if current_tile < 64:
                        raise
                    print(f"[Baikal] VRAM retry with LR tile {current_tile}")
                    model_management.soft_empty_cache()
        return (torch.stack(results),)
