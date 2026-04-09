import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import predict
import predict_onnx


def create_image(path, color="white"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


class PredictionScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "inputs")
        self.model_path = os.path.join(self.temp_dir.name, "best_model.pth")
        self.onnx_model_path = os.path.join(self.temp_dir.name, "best_model.onnx")

        create_image(os.path.join(self.input_dir, "root.jpg"), "red")
        create_image(os.path.join(self.input_dir, "nested", "child.png"), "blue")
        with open(self.model_path, "wb") as handle:
            handle.write(b"placeholder")
        with open(self.onnx_model_path, "wb") as handle:
            handle.write(b"placeholder")

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("predict.predict_single_image")
    @mock.patch("predict.load_torch_artifact", return_value={})
    @mock.patch("predict.get_data_transforms", return_value={"val": object()})
    @mock.patch("predict.get_device", return_value="cpu")
    @mock.patch("predict.setup_logging")
    def test_pytorch_prediction_recurses_into_subdirectories(
        self,
        _mock_setup_logging,
        _mock_get_device,
        _mock_get_data_transforms,
        _mock_load_torch_artifact,
        mock_predict_single_image,
    ):
        model = mock.MagicMock()
        with mock.patch("predict.get_orientation_model", return_value=model):
            predict.run_prediction(
                SimpleNamespace(
                    input_path=self.input_dir,
                    model_path=self.model_path,
                )
            )

        predicted_paths = [
            call.args[1]
            for call in mock_predict_single_image.call_args_list
        ]
        self.assertEqual(
            predicted_paths,
            [
                os.path.realpath(os.path.join(self.input_dir, "nested", "child.png")),
                os.path.realpath(os.path.join(self.input_dir, "root.jpg")),
            ],
        )

    @mock.patch("predict_onnx.predict_single_image_onnx")
    @mock.patch("predict_onnx.onnxruntime.InferenceSession")
    @mock.patch(
        "predict_onnx.onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    )
    @mock.patch("predict_onnx.get_data_transforms")
    @mock.patch("predict_onnx.setup_logging")
    def test_onnx_prediction_reuses_shared_val_transform_and_recurses(
        self,
        _mock_setup_logging,
        mock_get_data_transforms,
        _mock_get_available_providers,
        mock_inference_session,
        mock_predict_single_image_onnx,
    ):
        shared_transform = object()
        mock_get_data_transforms.return_value = {"val": shared_transform}
        mock_session = mock.MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_inference_session.return_value = mock_session

        predict_onnx.run_prediction_onnx(
            SimpleNamespace(
                input_path=self.input_dir,
                model_path=self.onnx_model_path,
            )
        )

        predicted_paths = [
            call.args[1]
            for call in mock_predict_single_image_onnx.call_args_list
        ]
        predicted_transforms = [
            call.args[2]
            for call in mock_predict_single_image_onnx.call_args_list
        ]

        self.assertEqual(
            predicted_paths,
            [
                os.path.realpath(os.path.join(self.input_dir, "nested", "child.png")),
                os.path.realpath(os.path.join(self.input_dir, "root.jpg")),
            ],
        )
        self.assertEqual(predicted_transforms, [shared_transform, shared_transform])


if __name__ == "__main__":
    unittest.main()
