import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from baikal.devices import resolve_device


class DeviceTests(unittest.TestCase):
    @patch("torch.cuda.current_device", return_value=2)
    @patch("torch.cuda.is_available", return_value=True)
    def test_unindexed_cuda_uses_current_gpu(self, available, current):
        self.assertEqual(resolve_device("cuda"), torch.device("cuda:2"))
        self.assertEqual(resolve_device(torch.device("cuda")), torch.device("cuda:2"))

    @patch("torch.cuda.current_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_explicit_gpu_is_preserved(self, available, current):
        self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
        current.assert_not_called()

    @patch("torch.cuda.current_device")
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_and_unavailable_cuda(self, available, current):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            resolve_device("cuda")
        current.assert_not_called()

    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.is_available", return_value=True)
    def test_wrapper_normalizes_before_comfy_and_reuses_patcher(self, available, current):
        management = SimpleNamespace(get_torch_device=lambda: torch.device("cuda"),
                                     unet_offload_device=lambda: torch.device("cpu"),
                                     load_models_gpu=Mock())
        factory = Mock(side_effect=lambda model, **kw: SimpleNamespace(**kw))
        comfy = SimpleNamespace(model_management=management,
                                model_patcher=SimpleNamespace(CoreModelPatcher=factory))
        spec = importlib.util.spec_from_file_location("baikal.wrapper", Path(__file__).parents[1] / "baikal/wrapper.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"comfy": comfy}):
            spec.loader.exec_module(module)
        model = SimpleNamespace(scale=2, window=8, stem=SimpleNamespace(out_channels=16), eval=Mock())
        wrapper = module.BaikalModel(model, "loopsr", "test.safetensors")
        wrapper.prepare(torch.device("cuda"), 64)
        wrapper.prepare(torch.device("cuda:0"), 64)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["load_device"], torch.device("cuda:0"))
        self.assertEqual(management.load_models_gpu.call_count, 2)
        gpu_patcher = wrapper.patcher
        wrapper.prepare("cpu", 64)
        wrapper.prepare("cuda", 64)
        self.assertIs(wrapper.patcher, gpu_patcher)
        self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
