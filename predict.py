import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import config
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def normalize_image_path(image_path: str) -> str:
    return os.path.normcase(os.path.realpath(image_path))


def discover_image_files(root_dir: str) -> list[str]:
    image_files = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(normalize_image_path(os.path.join(root, filename)))

    image_files.sort()
    if not image_files:
        raise ValueError(f"No images found in the directory: {root_dir}")

    return image_files


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA and MPS not available. Using CPU.")
    return device


def get_data_transforms() -> dict:
    return {
        "val": transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
                transforms.CenterCrop(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    }


def load_image_safely(path: str) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGB", "L"):
            return img.convert("RGB")

        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img)
        return background


def load_torch_artifact(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This project requires a PyTorch version that supports "
            "torch.load(..., weights_only=True)."
        ) from exc


def get_orientation_model(pretrained: bool = True):
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, config.NUM_CLASSES),
    )
    return model


def predict_single_image(model, image_path, device, transforms):
    """Predicts orientation for a single image file and logs the time taken."""

    start_time = time.time()  # Start timer

    try:
        image = load_image_safely(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    input_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    predicted_class = predicted_idx.item()
    result = config.CLASS_MAP[predicted_class]

    end_time = time.time()  # End timer
    duration = end_time - start_time

    print(
        f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result} (Took {duration:.4f} seconds)"
    )


def run_prediction(args):
    """Main prediction routine."""
    setup_logging()

    if not os.path.exists(args.model_path):
        logging.error(
            f"Model file not found at {args.model_path}. Please train the model first."
        )
        return

    device = get_device()
    all_transforms = get_data_transforms()
    transforms = all_transforms["val"]

    # Load the trained model
    model = get_orientation_model(pretrained=False)  # No need to download weights

    # Adjust state_dict keys if the model was compiled
    state_dict = load_torch_artifact(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image(model, input_path, device, transforms)
    elif os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        total_dir_start_time = time.time()  # Start timer for the entire directory
        try:
            image_files = discover_image_files(input_path)
        except ValueError:
            print(f"No image files found in directory: {input_path}")
            return

        for image_file in image_files:
            predict_single_image(model, image_file, device, transforms)

        total_dir_end_time = time.time()  # End timer
        total_duration = total_dir_end_time - total_dir_start_time
        print(
            f"Finished processing directory '{input_path}'. Total time: {total_duration:.4f} seconds for {len(image_files)} images."
        )
    else:
        print(f"Input path is not a valid file or directory: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image orientation.")
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

    args = parser.parse_args()
    run_prediction(args)
