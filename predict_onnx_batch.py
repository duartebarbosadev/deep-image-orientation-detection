import argparse
import logging
import os
import time

import numpy as np
import onnxruntime
import torch

import config
from src.dataset import discover_image_files
from src.utils import get_data_transforms, load_image_safely, setup_logging


PREFERRED_PROVIDERS = [
    "CUDAExecutionProvider",
    "MpsExecutionProvider",
    "ROCmExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def get_default_onnx_model_path() -> str:
    return os.path.join(config.MODEL_SAVE_DIR, "best_model.onnx")


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_input_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return discover_image_files(input_path)
    raise ValueError(f"Input path is not a valid file or directory: {input_path}")


def load_onnx_session(model_path: str):
    available_providers = onnxruntime.get_available_providers()
    logging.info(f"Available ONNX Runtime providers: {available_providers}")

    chosen_provider = next(
        (provider for provider in PREFERRED_PROVIDERS if provider in available_providers),
        "CPUExecutionProvider",
    )
    logging.info(f"Attempting to load ONNX model with provider: {chosen_provider}")

    ort_session = onnxruntime.InferenceSession(model_path, providers=[chosen_provider])
    actual_provider = ort_session.get_providers()[0]
    logging.info(
        f"Successfully loaded ONNX model from {model_path} using provider: {actual_provider}"
    )

    if chosen_provider != actual_provider and actual_provider == "CPUExecutionProvider":
        logging.warning(
            "Warning: ONNX Runtime fell back to CPU. "
            f"The chosen provider '{chosen_provider}' might not be correctly configured."
        )

    return ort_session


def build_input_batch(image_paths, transforms):
    valid_paths = []
    image_tensors = []

    for image_path in image_paths:
        try:
            image = load_image_safely(image_path)
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue
        except Exception as exc:
            print(f"Error opening image {image_path}: {exc}")
            continue

        image_tensors.append(transforms(image))
        valid_paths.append(image_path)

    if not image_tensors:
        return [], None

    return valid_paths, torch.stack(image_tensors).cpu().numpy()


def predict_image_batch_onnx(ort_session, image_paths, transforms):
    valid_paths, input_batch = build_input_batch(image_paths, transforms)
    if input_batch is None:
        return 0

    start_time = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: input_batch}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted_indices = np.argmax(ort_outs[0], axis=1).tolist()
    duration = time.time() - start_time

    for image_path, predicted_class in zip(valid_paths, predicted_indices):
        result = config.CLASS_MAP[predicted_class]
        print(
            f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result}"
        )

    print(
        f"Processed batch of {len(valid_paths)} image(s) in {duration:.4f} seconds."
    )
    return len(valid_paths)


def run_batch_prediction_onnx(args):
    setup_logging()

    if args.batch_size <= 0:
        logging.error("Batch size must be a positive integer.")
        return

    if not os.path.exists(args.model_path):
        logging.error(f"ONNX model file not found at {args.model_path}.")
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    try:
        image_files = collect_input_images(input_path)
    except ValueError as exc:
        print(exc)
        return

    transforms = get_data_transforms()["val"]

    try:
        ort_session = load_onnx_session(args.model_path)
    except Exception as exc:
        logging.error(f"Error loading ONNX model {args.model_path}: {exc}")
        logging.error(
            "If you are trying to use a GPU provider (CUDA, TensorRT, ROCm, MPS), "
            "please ensure the correct onnxruntime package is installed and drivers are up to date."
        )
        return

    if len(image_files) == 1 and os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
    else:
        print(
            f"Processing {len(image_files)} image(s) from directory in batches of {args.batch_size}: {input_path}"
        )

    total_dir_start_time = time.time()
    processed_count = 0
    for image_batch in iter_batches(image_files, args.batch_size):
        processed_count += predict_image_batch_onnx(ort_session, image_batch, transforms)

    total_duration = time.time() - total_dir_start_time
    print(
        f"Finished processing '{input_path}'. Total time: {total_duration:.4f} seconds for {processed_count} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict image orientation in batches using an ONNX model."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=get_default_onnx_model_path(),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to process per inference batch.",
    )

    run_batch_prediction_onnx(parser.parse_args())
