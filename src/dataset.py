import hashlib
import json
import logging
import os
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import config
from src.utils import (
    load_cached_image,
    load_image_safely,
    rotate_right_angle,
    validate_right_angle_rotations,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
CACHE_MANIFEST_FILENAME = "manifest.json"


def normalize_source_path(image_path: str) -> str:
    """Returns a canonical path for a source image."""
    return os.path.normcase(os.path.realpath(image_path))


def build_source_id(image_path: str) -> str:
    """Builds a stable identifier for a source image path."""
    normalized_path = normalize_source_path(image_path)
    return hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()


def discover_upright_image_files(upright_dir: str) -> list[str]:
    """Recursively discovers supported images under an upright image directory."""
    image_files = []
    for root, _, files in os.walk(upright_dir):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(normalize_source_path(os.path.join(root, filename)))

    image_files.sort()
    if not image_files:
        raise ValueError(f"No images found in the directory: {upright_dir}")

    return image_files


def split_image_files(
    image_files: list[str], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Splits source images into reproducible train and validation lists."""
    if not image_files:
        raise ValueError("Cannot split an empty image list.")
    if len(image_files) < 2:
        raise ValueError("At least 2 source images are required for a train/val split.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    shuffled_files = list(image_files)
    random.Random(seed).shuffle(shuffled_files)

    train_size = int(len(shuffled_files) * train_ratio)
    train_size = max(1, min(train_size, len(shuffled_files) - 1))

    return shuffled_files[:train_size], shuffled_files[train_size:]


def get_cache_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CACHE_MANIFEST_FILENAME)


def load_cache_manifest(cache_dir: str) -> list[dict]:
    """Loads the cached dataset manifest."""
    manifest_path = get_cache_manifest_path(cache_dir)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Cache manifest does not exist: '{manifest_path}'. Rebuild the cache."
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Cache manifest is empty or invalid: '{manifest_path}'.")

    return manifest


# Dataset for cases where caching is not desired
class ImageOrientationDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = [normalize_source_path(path) for path in image_files]
        if not self.image_files:
            raise ValueError("No image files were provided to the dataset.")
        self.transform = transform
        self.rotations = config.ROTATIONS
        validate_right_angle_rotations(self.rotations)
        self.num_rotations = len(self.rotations)

    def __len__(self):
        return len(self.image_files) * self.num_rotations

    def __getitem__(self, idx):
        image_idx = idx // self.num_rotations
        label = idx % self.num_rotations

        image_path = self.image_files[image_idx]
        angle_to_rotate = self.rotations[label]
        image = load_image_safely(image_path)
        rotated_image = rotate_right_angle(image, angle_to_rotate)

        if self.transform:
            image_tensor = self.transform(rotated_image)
        else:
            # Default minimal transformation if none provided
            image_tensor = transforms.ToTensor()(rotated_image)

        return image_tensor, torch.tensor(label, dtype=torch.long)


# This dataset reads directly from the pre-processed and cached images.
# This is significantly faster (if run on a fast disk) as it only has to do a file read and basic tensor conversion.
class ImageOrientationDatasetFromCache(Dataset):
    def __init__(self, cache_dir, source_ids=None, samples=None, transform=None):
        self.cache_dir = cache_dir
        self.transform = transform

        if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
            raise FileNotFoundError(
                f"Cache directory is empty or does not exist: '{cache_dir}'. "
                "Run the caching process in `train.py` first."
            )

        if (source_ids is None) == (samples is None):
            raise ValueError("Provide exactly one of source_ids or samples.")

        if samples is None:
            source_id_set = set(source_ids)
            manifest = load_cache_manifest(cache_dir)
            manifest_source_ids = {
                sample["source_id"] for sample in manifest if "source_id" in sample
            }
            missing_source_ids = source_id_set - manifest_source_ids
            if missing_source_ids:
                logging.warning(
                    "Cache manifest is missing %d requested source IDs. "
                    "Matched %d of %d requested source images.",
                    len(missing_source_ids),
                    len(source_id_set) - len(missing_source_ids),
                    len(source_id_set),
                )
            samples = [
                sample for sample in manifest if sample.get("source_id") in source_id_set
            ]

        self.samples = sorted(samples, key=lambda sample: sample["cached_path"])

        if not self.samples:
            raise ValueError("No cached samples matched the requested source images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["cached_path"]
        label = int(sample["label"])
        image = load_cached_image(image_path)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, torch.tensor(label, dtype=torch.long)
