import os
import unittest

import config
from convert_to_onnx import get_default_onnx_output_path
from predict_onnx import get_default_onnx_model_path


class OnnxPathTests(unittest.TestCase):
    def test_export_default_output_path_uses_checkpoint_basename(self):
        self.assertEqual(
            get_default_onnx_output_path(os.path.join("models", "best_model.pth")),
            os.path.join("models", "best_model.onnx"),
        )

    def test_predict_default_path_matches_default_exported_best_model(self):
        self.assertEqual(
            get_default_onnx_model_path(),
            os.path.join(config.MODEL_SAVE_DIR, "best_model.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
