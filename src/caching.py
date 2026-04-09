import json
import logging
import os
from multiprocessing import Pool, cpu_count

from PIL import Image

import config
from tqdm import tqdm

from src.dataset import (
    build_source_id,
    discover_upright_image_files,
    get_cache_manifest_path,
    normalize_source_path,
)
from src.utils import load_image_safely, rotate_right_angle


def process_and_cache_image(args):
    """
    Worker function to process a single image. It uses the robust loader,
    creates four rotated versions, and saves them to the cache directory.
    """
    image_path, cache_dir = args
    try:
        source_path = normalize_source_path(image_path)
        source_id = build_source_id(source_path)

        # Use the single, robust image loader from utils
        img = load_image_safely(source_path)
        manifest_entries = []

        # Use the rotation definition from config
        for label, angle in config.ROTATIONS.items():
            rotated_img = rotate_right_angle(img, angle)
            cached_filename = f"{source_id}__{label}.png"
            save_path = os.path.join(cache_dir, cached_filename)
            rotated_img.save(
                save_path,
                "PNG",
            )
            manifest_entries.append(
                {
                    "source_path": source_path,
                    "source_id": source_id,
                    "cached_path": os.path.abspath(save_path),
                    "label": label,
                }
            )

        return {"entries": manifest_entries, "failure": None}

    except Exception as e:
        logging.warning(f"Could not process and cache {image_path}. Error: {e}")
        return {"entries": [], "failure": image_path}


def cache_dataset(upright_dir=None, num_workers=None, force_rebuild=False):
    """
    Applies rotations to all images and saves them to a cache, using
    multiple processes.
    """
    upright_dir = upright_dir or config.DATA_DIR
    cache_dir = config.CACHE_DIR
    manifest_path = get_cache_manifest_path(cache_dir)

    if not os.path.exists(upright_dir):
        logging.error(f"Source data directory not found: {upright_dir}")
        raise FileNotFoundError(f"Source data directory not found: {upright_dir}")

    os.makedirs(cache_dir, exist_ok=True)

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".png")]
    if force_rebuild:
        logging.info(
            f"Force rebuild is True. Clearing {len(cached_files)} cached files from cache directory: {cache_dir}"
        )
        for f in cached_files:
            os.remove(os.path.join(cache_dir, f))
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        cached_files = []

    if cached_files and os.path.exists(manifest_path):
        logging.info(
            f"Cache already exists with {len(cached_files)} files at '{cache_dir}'. Skipping rebuild."
        )
        return
    if cached_files or os.path.exists(manifest_path):
        logging.info("Cache contents and manifest are out of sync. Rebuilding cache...")
        for f in cached_files:
            os.remove(os.path.join(cache_dir, f))
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    logging.info("Cache is empty or was cleared. Starting build process...")

    image_files = discover_upright_image_files(upright_dir)

    configured_workers = config.NUM_WORKERS if num_workers is None else num_workers
    num_workers = configured_workers if configured_workers > 0 else cpu_count()
    chunk_size = max(1, len(image_files) // max(1, num_workers * 8))
    logging.info(
        f"Building cache with {num_workers} worker processes (chunksize={chunk_size})..."
    )

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    process_and_cache_image,
                    [(image_path, cache_dir) for image_path in image_files],
                    chunksize=chunk_size,
                ),
                total=len(image_files),
                desc="Caching Images",
            )
        )

    manifest_entries = []
    failures = []
    for result in results:
        manifest_entries.extend(result["entries"])
        if result["failure"] is not None:
            failures.append(result["failure"])

    if failures:
        logging.warning(
            f"Warning: {len(failures)} out of {len(image_files)} images failed to process. Check logs for details."
        )

    manifest_entries.sort(key=lambda entry: (entry["source_path"], entry["label"]))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_entries, handle, indent=2)

    cached_file_count = len([f for f in os.listdir(cache_dir) if f.endswith(".png")])
    logging.info(
        f"Successfully built image cache with {cached_file_count} files."
    )
