import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

import predict_batch
import predict_onnx_batch


def create_image(path, color="white"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


class DummyTorchBatchModel:
    def __init__(self):
        self.batch_sizes = []

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, inputs):
        self.batch_sizes.append(inputs.shape[0])
        return torch.zeros((inputs.shape[0], 4), dtype=torch.float32)


class BatchPredictionScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "inputs")
        self.model_path = os.path.join(self.temp_dir.name, "best_model.pth")
        self.onnx_model_path = os.path.join(self.temp_dir.name, "best_model.onnx")

        create_image(os.path.join(self.input_dir, "root.jpg"), "red")
        create_image(os.path.join(self.input_dir, "nested", "child.png"), "blue")
        create_image(os.path.join(self.input_dir, "nested", "child_2.jpeg"), "green")

        with open(self.model_path, "wb") as handle:
            handle.write(b"placeholder")
        with open(self.onnx_model_path, "wb") as handle:
            handle.write(b"placeholder")

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("predict_batch.get_data_transforms")
    @mock.patch("predict_batch.get_device", return_value=torch.device("cpu"))
    @mock.patch("predict_batch.setup_logging")
    @mock.patch("predict_batch.load_torch_artifact", return_value={})
    def test_pytorch_batch_script_processes_directory_in_batches(
        self,
        _mock_load_torch_artifact,
        _mock_setup_logging,
        _mock_get_device,
        mock_get_data_transforms,
    ):
        mock_get_data_transforms.return_value = {"val": lambda image: torch.ones(3, 4, 4)}
        model = DummyTorchBatchModel()

        with mock.patch("predict_batch.get_orientation_model", return_value=model):
            predict_batch.run_batch_prediction(
                SimpleNamespace(
                    input_path=self.input_dir,
                    model_path=self.model_path,
                    batch_size=2,
                )
            )

        self.assertEqual(model.batch_sizes, [2, 1])

    @mock.patch("predict_onnx_batch.setup_logging")
    @mock.patch("predict_onnx_batch.get_data_transforms")
    @mock.patch(
        "predict_onnx_batch.onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    )
    @mock.patch("predict_onnx_batch.onnxruntime.InferenceSession")
    def test_onnx_batch_script_processes_directory_in_batches(
        self,
        mock_inference_session,
        _mock_get_available_providers,
        mock_get_data_transforms,
        _mock_setup_logging,
    ):
        mock_get_data_transforms.return_value = {"val": lambda image: torch.ones(3, 4, 4)}
        session_batch_sizes = []
        mock_session = mock.MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.get_inputs.return_value = [SimpleNamespace(name="input")]

        def run_side_effect(_outputs, ort_inputs):
            batch = ort_inputs["input"]
            session_batch_sizes.append(batch.shape[0])
            return [torch.zeros((batch.shape[0], 4), dtype=torch.float32).numpy()]

        mock_session.run.side_effect = run_side_effect
        mock_inference_session.return_value = mock_session

        predict_onnx_batch.run_batch_prediction_onnx(
            SimpleNamespace(
                input_path=self.input_dir,
                model_path=self.onnx_model_path,
                batch_size=2,
            )
        )

        self.assertEqual(session_batch_sizes, [2, 1])


if __name__ == "__main__":
    unittest.main()
