import unittest

import torch

from src.utils import get_amp_autocast_kwargs, get_grad_scaler, should_use_channels_last


class DeviceUtilsTests(unittest.TestCase):
    def test_cuda_device_enables_bfloat16_autocast(self):
        self.assertEqual(
            get_amp_autocast_kwargs(torch.device("cuda")),
            {"device_type": "cuda", "dtype": torch.bfloat16},
        )

    def test_cpu_device_uses_fp32_training(self):
        self.assertIsNone(get_amp_autocast_kwargs(torch.device("cpu")))

    def test_mps_device_enables_float16_autocast(self):
        self.assertEqual(
            get_amp_autocast_kwargs(torch.device("mps")),
            {"device_type": "mps", "dtype": torch.float16},
        )

    def test_grad_scaler_is_enabled_for_mps(self):
        self.assertTrue(get_grad_scaler(torch.device("mps")).is_enabled())

    def test_grad_scaler_is_disabled_for_cpu(self):
        self.assertFalse(get_grad_scaler(torch.device("cpu")).is_enabled())

    def test_channels_last_is_enabled_for_mps(self):
        self.assertTrue(should_use_channels_last(torch.device("mps")))

    def test_channels_last_is_disabled_for_cpu(self):
        self.assertFalse(should_use_channels_last(torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
