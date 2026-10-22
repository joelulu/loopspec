"""Backend selection tests without CUDA, FlashInfer, or a PyTorch install."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


def load_backend():
    torch = ModuleType('torch')
    torch.Tensor = type('Tensor', (), {})
    torch.device = type('device', (), {})
    torch.cuda = Mock()
    path = Path(__file__).resolve().parents[1] / 'sglang_recurrent/integrations/flashinfer.py'
    spec = importlib.util.spec_from_file_location('isolated_linear_backend', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch}):
        spec.loader.exec_module(module)
    return module, torch


class BackendSelectionTests(unittest.TestCase):
    def test_torch_skips_cuda_and_flashinfer(self):
        module, torch = load_backend()
        # A missing/broken FlashInfer must never be imported in explicit torch mode.
        with patch.dict(os.environ, {'LOOPSPEC_LINEAR_BACKEND': 'torch'}), \
             patch.dict(sys.modules, {'flashinfer': None, 'flashinfer.gemm': None}):
            self.assertEqual(module.install_linear_backend()['active'], 'torch')
        self.assertEqual(torch.cuda.mock_calls, [])

    def test_invalid_setting_fails_before_cuda(self):
        module, torch = load_backend()
        with patch.dict(os.environ, {'LOOPSPEC_LINEAR_BACKEND': 'typo'}):
            with self.assertRaises(ValueError):
                module.install_linear_backend()
        self.assertEqual(torch.cuda.mock_calls, [])

    def test_cannot_report_torch_after_patch_installed(self):
        module, _ = load_backend()
        module._installed = True
        with patch.dict(os.environ, {'LOOPSPEC_LINEAR_BACKEND': 'torch'}):
            with self.assertRaises(RuntimeError):
                module.install_linear_backend()

    def test_auto_retains_no_cuda_fallback(self):
        module, torch = load_backend()
        torch.cuda.is_available.return_value = False
        with patch.dict(os.environ, {'LOOPSPEC_LINEAR_BACKEND': 'auto'}):
            self.assertEqual(module.install_linear_backend()['reason'], 'CUDA is unavailable')


if __name__ == '__main__':
    unittest.main()
