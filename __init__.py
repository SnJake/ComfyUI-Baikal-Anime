from .baikal.nodes import BaikalModelLoader, BaikalUpscale

# Stable internal IDs keep saved workflows loadable after the rebrand.
NODE_CLASS_MAPPINGS = {
    "SnJakeAnimeUpscaleCheckpointLoader": BaikalModelLoader,
    "SnJakeAnimeUpscaleInference": BaikalUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SnJakeAnimeUpscaleCheckpointLoader": "Baikal Model Loader",
    "SnJakeAnimeUpscaleInference": "Baikal Anime Upscale",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
