import unittest
from unittest.mock import patch

import torch

from src.utils import (
    get_amp_autocast_kwargs,
    get_grad_scaler,
    move_batch_to_device,
    should_use_channels_last,
)


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

    def test_grad_scaler_is_disabled_for_cuda_bfloat16(self):
        self.assertFalse(get_grad_scaler(torch.device("cuda")).is_enabled())

    def test_grad_scaler_is_disabled_for_cpu(self):
        self.assertFalse(get_grad_scaler(torch.device("cpu")).is_enabled())

    def test_channels_last_is_disabled_for_mps(self):
        self.assertFalse(should_use_channels_last(torch.device("mps")))

    def test_channels_last_is_disabled_for_cpu(self):
        self.assertFalse(should_use_channels_last(torch.device("cpu")))

    def test_move_batch_to_device_uses_channels_last_for_4d_inputs(self):
        inputs = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 1])

        with patch("src.utils.should_use_channels_last", return_value=True):
            moved_inputs, moved_labels = move_batch_to_device(
                inputs, labels, torch.device("cpu")
            )

        self.assertTrue(
            moved_inputs.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(torch.equal(moved_labels, labels))

    def test_move_batch_to_device_keeps_default_layout_for_non_4d_inputs(self):
        inputs = torch.randn(2, 8, 8)
        labels = torch.tensor([0, 1])

        with patch("src.utils.should_use_channels_last", return_value=True):
            moved_inputs, moved_labels = move_batch_to_device(
                inputs, labels, torch.device("cpu")
            )

        self.assertTrue(moved_inputs.is_contiguous())
        self.assertTrue(torch.equal(moved_labels, labels))


if __name__ == "__main__":
    unittest.main()
