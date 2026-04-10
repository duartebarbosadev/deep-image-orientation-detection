import unittest
from unittest.mock import patch

from src.utils import load_torch_artifact


class TorchLoadSafetyTests(unittest.TestCase):
    @patch("src.utils.torch.load", return_value={"ok": True})
    def test_load_torch_artifact_enforces_weights_only(self, mock_torch_load):
        result = load_torch_artifact("checkpoint.pth", map_location="cpu")

        self.assertEqual(result, {"ok": True})
        mock_torch_load.assert_called_once_with(
            "checkpoint.pth",
            map_location="cpu",
            weights_only=True,
        )


if __name__ == "__main__":
    unittest.main()
