import unittest
from unittest import mock

import torch
import torch.nn as nn

import src.model as model_module


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class DummyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([DummyBlock() for _ in range(4)])
        self.classifier = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(1, 1))


class ModelConfigTests(unittest.TestCase):
    def test_get_orientation_model_uses_config_default_unfreeze_count(self):
        dummy_model = DummyEfficientNet()

        with mock.patch.object(
            model_module.config, "NUM_BLOCKS_TO_UNFREEZE", 1
        ), mock.patch(
            "src.model.models.efficientnet_v2_s",
            return_value=dummy_model,
        ):
            model_module.get_orientation_model(pretrained=False)

        self.assertFalse(dummy_model.features[0].weight.requires_grad)
        self.assertFalse(dummy_model.features[1].weight.requires_grad)
        self.assertFalse(dummy_model.features[2].weight.requires_grad)
        self.assertTrue(dummy_model.features[3].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
