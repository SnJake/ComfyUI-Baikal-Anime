import unittest

import torch
from torch.nn import functional as F

from training_code.loop_sr.losses import AnimeLoss, gradients, luminance, multiscale_edge_loss


class MultiscaleEdgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_default_preserves_v1_edge_exactly(self):
        target = torch.rand(2, 3, 17, 23)
        prediction = torch.rand_like(target)
        pdx, pdy = gradients(luminance(prediction))
        tdx, tdy = gradients(luminance(target))
        expected = ((pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()) / 2
        _, details = AnimeLoss()(prediction, target)
        self.assertTrue(torch.equal(details["edge"], expected))

    def test_scale_normalization_keeps_linear_ramp_strength(self):
        target = torch.full((1, 3, 16, 16), 0.2)
        prediction = target + torch.arange(16).view(1, 1, 1, 16) * 0.02
        single = multiscale_edge_loss(prediction, target, (1.0,))
        both = multiscale_edge_loss(prediction, target, (1.0, 0.5))
        self.assertAlmostEqual(single.item(), 0.01, places=7)
        self.assertAlmostEqual(both.item(), 0.01, places=7)

    def test_half_scale_filters_checkerboard_before_gradient_comparison(self):
        grid = (torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2
        target = torch.full((1, 3, 16, 16), 0.5)
        prediction = target + (grid * 2 - 1) * 0.04
        self.assertAlmostEqual(multiscale_edge_loss(prediction, target, (1.0,)).item(), 0.08, places=6)
        self.assertAlmostEqual(multiscale_edge_loss(prediction, target, (0.5,)).item(), 0.0, places=7)
        self.assertAlmostEqual(multiscale_edge_loss(prediction, target, (1.0, 0.5)).item(), 0.04, places=6)

    def test_signed_edges_prefer_correct_target_to_blur_and_reversed_contrast(self):
        target = torch.full((1, 3, 32, 32), 0.2)
        target[..., 16:] = 0.8
        blurred = F.avg_pool2d(F.pad(target, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        criterion = AnimeLoss(edge_scales=[1.0, 0.5])
        correct, clean = criterion(target, target)
        _, blur = criterion(blurred, target)
        _, reverse = criterion(1 - target, target)
        self.assertEqual(correct.item(), 0)
        self.assertLess(clean["edge"].item(), blur["edge"].item())
        self.assertLess(clean["edge"].item(), reverse["edge"].item())

    def test_deep_supervision_gradients_and_microbatch_partition(self):
        target = torch.rand(2, 3, 17, 23)
        outputs = [(target + torch.randn_like(target) * 0.02).requires_grad_() for _ in range(4)]
        criterion = AnimeLoss(edge_scales=[1.0, 0.5])
        together, details = criterion(outputs, target)
        separated = torch.stack([criterion([p[i:i+1] for p in outputs], target[i:i+1])[0] for i in range(2)]).mean()
        self.assertTrue(torch.allclose(together, separated, atol=1e-7))
        grad_a = torch.autograd.grad(together, outputs, retain_graph=True)
        grad_b = torch.autograd.grad(separated, outputs)
        for a, b in zip(grad_a, grad_b):
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(a.abs().sum().item(), 0)
            self.assertTrue(torch.allclose(a, b, atol=1e-7))
        expected = criterion(outputs[-1], target)[0] + torch.stack([criterion(p, target)[0] for p in outputs[:-1]]).mean()
        self.assertTrue(torch.allclose(together, expected, atol=1e-7))
        self.assertEqual(details["edge"].dtype, torch.float32)

    def test_small_shapes_bf16_and_invalid_scales(self):
        criterion = AnimeLoss(edge_scales=[1.0, 0.5])
        for h, w in ((1, 1), (1, 7), (3, 5), (9, 11)):
            with self.subTest(shape=(h, w)):
                target = torch.rand(1, 3, h, w)
                prediction = torch.rand_like(target).to(torch.bfloat16).requires_grad_()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss, _ = criterion(prediction, target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(prediction.grad).all())
        for scales in ([], [0], [0.25], [2]):
            with self.assertRaisesRegex(ValueError, "edge_scales"):
                AnimeLoss(edge_scales=scales)


if __name__ == "__main__":
    unittest.main()
