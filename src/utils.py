import torch
import logging
import sys
from contextlib import nullcontext
import torchvision.transforms as transforms
from config import IMAGE_SIZE
from PIL import Image, ImageOps


RIGHT_ANGLE_ROTATIONS = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_device() -> torch.device:
    """
    Selects the best available device (CUDA, MPS, or CPU) and returns it.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA and MPS not available. Using CPU.")
    return device


def get_amp_autocast_kwargs(device: torch.device):
    """
    Returns autocast settings for the selected device, or None when training
    should run in standard FP32.
    """
    if device.type == "cuda":
        return {"device_type": "cuda", "dtype": torch.bfloat16}
    if device.type == "mps":
        return {"device_type": "mps", "dtype": torch.float16}
    return None


def get_autocast_context(device: torch.device):
    """
    Returns a device-aware autocast context manager.
    CUDA uses bfloat16 mixed precision, MPS uses float16 mixed precision,
    and CPU stays in FP32.
    """
    autocast_kwargs = get_amp_autocast_kwargs(device)
    if autocast_kwargs is None:
        return nullcontext()
    return torch.amp.autocast(**autocast_kwargs)


def get_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """
    Returns a GradScaler only when the configured AMP dtype uses float16.
    """
    autocast_kwargs = get_amp_autocast_kwargs(device)
    autocast_dtype = None if autocast_kwargs is None else autocast_kwargs.get("dtype")
    use_grad_scaler = autocast_dtype == torch.float16
    return torch.amp.GradScaler(device=device.type, enabled=use_grad_scaler)


def should_use_channels_last(device: torch.device) -> bool:
    """
    Convolution-heavy models run faster with channels-last tensors on CUDA/cuDNN.
    MPS is excluded: EfficientNet's internal .view() calls are incompatible with
    the non-contiguous memory layout channels-last produces on MPS.
    """
    return device.type == "cuda"


def move_batch_to_device(inputs, labels, device: torch.device, non_blocking: bool = False):
    """
    Moves a batch to the selected device and applies channels-last layout on
    GPU backends to improve convolution throughput.
    """
    if should_use_channels_last(device) and getattr(inputs, "ndim", 0) == 4:
        inputs = inputs.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    else:
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    labels = labels.to(device=device, non_blocking=non_blocking)
    return inputs, labels


def get_data_transforms() -> dict:
    """
    Returns a dictionary of data transformations for training and validation.
    """
    return {
        "train": transforms.Compose(
            [
                # Use a crop that preserves more of the image center
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                # ColorJitter is a good augmentation that doesn't affect orientation
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                # RandomErasing is also a good regularizer
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            ]
        ),
        "val": transforms.Compose(
            [
                # Validation transform is fine as is
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }


def load_image_safely(path: str) -> Image.Image:
    """
    Loads an image, respects EXIF orientation, and safely converts it to a
    3-channel RGB format. It handles palletized images and images with
    transparency by compositing them onto a white background. This is the
    most robust way to prevent processing errors.
    """
    with Image.open(path) as img:
        # Respect the EXIF orientation tag before any other processing.
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGB", "L"):
            return img.convert("RGB")

        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img)
        return background


def load_cached_image(path: str) -> Image.Image:
    """
    Loads a cached training image quickly.
    Cached files are generated by this project, so they do not need EXIF
    correction or alpha compositing.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def rotate_right_angle(image: Image.Image, angle: int) -> Image.Image:
    """
    Rotates an image using exact 90-degree transposes instead of generic
    resampling. This is faster and lossless for right-angle rotations.
    """
    normalized_angle = angle % 360
    if normalized_angle not in RIGHT_ANGLE_ROTATIONS:
        raise ValueError(f"Unsupported rotation angle: {angle}")
    transpose_op = RIGHT_ANGLE_ROTATIONS[normalized_angle]
    if transpose_op is None:
        return image
    return image.transpose(transpose_op)


def validate_right_angle_rotations(rotations: dict) -> None:
    """
    Ensures a rotation configuration only contains supported right angles.
    """
    invalid_rotations = [
        (label, angle)
        for label, angle in rotations.items()
        if angle % 360 not in RIGHT_ANGLE_ROTATIONS
    ]
    if invalid_rotations:
        formatted_rotations = ", ".join(
            f"label {label}: {angle}" for label, angle in invalid_rotations
        )
        raise ValueError(
            "Unsupported rotation angles in config.ROTATIONS: "
            f"{formatted_rotations}"
        )
