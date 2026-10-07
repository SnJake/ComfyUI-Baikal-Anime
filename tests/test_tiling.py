import unittest
import torch
from torch.nn import functional as F
from baikal.tiling import upscale
from training_code.loop_sr.model import LoopedSR


class TileTests(unittest.TestCase):
    def test_odd_tiny_and_batch_geometry(self):
        for h, w in [(1, 1), (17, 29), (35, 47)]:
            image = torch.rand(1, 3, h, w)
            result = upscale(lambda x: F.interpolate(x, scale_factor=2, mode="nearest"),
                             image, 2, 8, 16, 8, 8, lambda: None)
            self.assertEqual(result.shape, (1, 3, h*2, w*2))
            self.assertTrue(torch.allclose(result, F.interpolate(image, scale_factor=2, mode="nearest"), atol=1e-6))

    def test_full_context_tiling_and_early_exit(self):
        torch.set_num_threads(2)
        model = LoopedSR(dim=16, heads=2, pre_blocks=1, loop_blocks=2, post_blocks=1, ffn_ratio=1.5).eval()
        image = torch.rand(1, 3, 17, 29)
        for loops in [1, 4]:
            call = lambda x: model(x, loops=loops)
            whole = upscale(call, image, 2, 8, 0, 0, 0, lambda: None)
            tiled = upscale(call, image, 2, 8, 16, 8, 64, lambda: None)
            self.assertTrue(torch.allclose(whole, tiled, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
