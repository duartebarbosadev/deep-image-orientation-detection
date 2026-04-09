import torch
import os
import argparse
import logging
import time
import onnxruntime

import config
from src.utils import get_data_transforms, load_image_safely, setup_logging
from src.dataset import discover_image_files


def get_default_onnx_model_path() -> str:
    """Returns the ONNX path produced by converting the default best checkpoint."""
    return os.path.join(config.MODEL_SAVE_DIR, "best_model.onnx")


def predict_single_image_onnx(ort_session, image_path, image_transforms):
    """Predicts orientation for a single image file using the ONNX model and logs the time taken."""

    start_time = time.time()  # Start timer

    try:
        image = load_image_safely(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # Apply transformations and convert to NumPy array for ONNX Runtime
    input_tensor = image_transforms(image).unsqueeze(0).cpu().numpy()

    # Get input name from the ONNX session
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}

    # Run inference
    ort_outs = ort_session.run(None, ort_inputs)

    # Process output: ONNX Runtime returns a list of outputs, we need the first one
    output = torch.from_numpy(ort_outs[0])
    _, predicted_idx = torch.max(output, 1)

    predicted_class = predicted_idx.item()
    result = config.CLASS_MAP[predicted_class]

    end_time = time.time()  # End timer
    duration = end_time - start_time

    print(
        f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result} (Took {duration:.4f} seconds)"
    )


def run_prediction_onnx(args):
    """Main ONNX prediction routine."""
    setup_logging()

    if not os.path.exists(args.model_path):
        logging.error(f"ONNX model file not found at {args.model_path}.")
        return

    image_transforms = get_data_transforms()["val"]

    # Define a priority list for execution providers.
    # ONNX Runtime will try to use the first one in this list that is available on the system.
    PREFERRED_PROVIDERS = [
        #'TensorrtExecutionProvider',  # For NVIDIA GPUs with TensorRT (commenting out for now)
        "CUDAExecutionProvider",  # For NVIDIA GPUs
        "MpsExecutionProvider",  # For Apple Silicon (M1/M2/M3) GPUs
        "ROCmExecutionProvider",  # For AMD GPUs
        "CoreMLExecutionProvider",  # For Apple devices (can use Neural Engine)
        "CPUExecutionProvider",  # Universal fallback
    ]

    try:
        available_providers = onnxruntime.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {available_providers}")

        chosen_provider = None
        for provider in PREFERRED_PROVIDERS:
            if provider in available_providers:
                chosen_provider = provider
                break

        if not chosen_provider:
            logging.warning(
                "No preferred provider found. Defaulting to CPUExecutionProvider. "
                "This should not happen as CPUExecutionProvider is always available."
            )
            chosen_provider = "CPUExecutionProvider"

        logging.info(f"Attempting to load ONNX model with provider: {chosen_provider}")

        # Load the ONNX model with the single, highest-priority available provider
        ort_session = onnxruntime.InferenceSession(
            args.model_path, providers=[chosen_provider]
        )

        actual_provider = ort_session.get_providers()[0]
        logging.info(
            f"Successfully loaded ONNX model from {args.model_path} using provider: {actual_provider}"
        )

        if (
            chosen_provider != actual_provider
            and actual_provider == "CPUExecutionProvider"
        ):
            logging.warning(
                f"Warning: ONNX Runtime fell back to CPU. The chosen provider '{chosen_provider}' might not be correctly configured."
            )

    except Exception as e:
        logging.error(f"Error loading ONNX model {args.model_path}: {e}")
        logging.error(
            "If you are trying to use a GPU provider (CUDA, TensorRT, ROCm, MPS), "
            "please ensure the correct onnxruntime package is installed and drivers are up to date."
        )
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image_onnx(ort_session, input_path, image_transforms)
    elif os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        total_dir_start_time = time.time()  # Start timer for the entire directory
        try:
            image_files = discover_image_files(input_path)
        except ValueError:
            print(f"No image files found in directory: {input_path}")
            return

        for image_file in image_files:
            predict_single_image_onnx(ort_session, image_file, image_transforms)

        total_dir_end_time = time.time()  # End timer
        total_duration = total_dir_end_time - total_dir_start_time
        print(
            f"Finished processing directory '{input_path}'. Total time: {total_duration:.4f} seconds for {len(image_files)} images."
        )
    else:
        print(f"Input path is not a valid file or directory: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict image orientation using an ONNX model."
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

    args = parser.parse_args()
    run_prediction_onnx(args)
