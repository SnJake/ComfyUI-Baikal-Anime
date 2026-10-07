"""Only the two supported releases are exposed in the UI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "SnJake/Baikal-Anime-Upscaler"
MODELS = {
    "Baikal_LoopSR_x2.safetensors": {"kind": "loopsr", "config": ROOT / "configs/loopsr_x2.json"},
    "Baikal_SwinFIR_Anime_x2_v31.safetensors": {"kind": "swinfir", "config": ROOT / "configs/swinfir_v31.json"},
}
