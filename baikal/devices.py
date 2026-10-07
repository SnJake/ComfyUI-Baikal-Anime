import torch


def resolve_device(device):
    """Give CUDA an explicit index for ComfyUI's dynamic VRAM loader."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose CPU")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device
