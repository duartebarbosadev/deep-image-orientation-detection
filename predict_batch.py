import argparse
import logging
import os
import time

import torch

import config
from src.dataset import discover_image_files
from src.model import get_orientation_model
from src.utils import (
    get_data_transforms,
    get_device,
    load_image_safely,
    load_torch_artifact,
    setup_logging,
)


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_input_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return discover_image_files(input_path)
    raise ValueError(f"Input path is not a valid file or directory: {input_path}")


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

    return valid_paths, torch.stack(image_tensors)


def predict_image_batch(model, image_paths, device, transforms):
    valid_paths, input_batch = build_input_batch(image_paths, transforms)
    if input_batch is None:
        return 0

    start_time = time.time()
    input_batch = input_batch.to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        predicted_indices = torch.argmax(outputs, dim=1).tolist()
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


def run_batch_prediction(args):
    setup_logging()

    if args.batch_size <= 0:
        logging.error("Batch size must be a positive integer.")
        return

    if not os.path.exists(args.model_path):
        logging.error(
            f"Model file not found at {args.model_path}. Please train the model first."
        )
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

    device = get_device()
    transforms = get_data_transforms()["val"]

    model = get_orientation_model(pretrained=False)
    state_dict = load_torch_artifact(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if len(image_files) == 1 and os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
    else:
        print(
            f"Processing {len(image_files)} image(s) from directory in batches of {args.batch_size}: {input_path}"
        )

    total_dir_start_time = time.time()
    processed_count = 0
    for image_batch in iter_batches(image_files, args.batch_size):
        processed_count += predict_image_batch(model, image_batch, device, transforms)

    total_duration = time.time() - total_dir_start_time
    print(
        f"Finished processing '{input_path}'. Total time: {total_duration:.4f} seconds for {processed_count} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict image orientation in batches."
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
        default=os.path.join(config.MODEL_SAVE_DIR, "best_model.pth"),
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to process per inference batch.",
    )

    run_batch_prediction(parser.parse_args())
