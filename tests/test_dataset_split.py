import json
import os
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

from PIL import Image

import config
import train as train_module
from src.dataset import (
    ImageOrientationDataset,
    ImageOrientationDatasetFromCache,
    build_source_id,
    discover_upright_image_files,
    get_cache_manifest_path,
    split_image_files,
)


def create_image(path, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "upright_images")
        self.cache_dir = os.path.join(self.temp_dir.name, "cache")

        self.image_paths = [
            os.path.join(self.data_dir, "group_a", "sample_1.jpg"),
            os.path.join(self.data_dir, "group_a", "sample_2.jpg"),
            os.path.join(self.data_dir, "group_b", "sample_3.png"),
            os.path.join(self.data_dir, "group_b", "sample_4.jpeg"),
            os.path.join(self.data_dir, "group_c", "sample_5.jpg"),
        ]

        colors = ["red", "green", "blue", "yellow", "purple"]
        for image_path, color in zip(self.image_paths, colors):
            create_image(image_path, color)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_helper_is_reproducible_and_disjoint(self):
        discovered_files = discover_upright_image_files(self.data_dir)

        train_files, val_files = split_image_files(discovered_files, seed=7)
        train_files_repeat, val_files_repeat = split_image_files(discovered_files, seed=7)

        self.assertEqual(train_files, train_files_repeat)
        self.assertEqual(val_files, val_files_repeat)
        self.assertEqual(set(discovered_files), set(train_files) | set(val_files))
        self.assertTrue(set(train_files).isdisjoint(val_files))

    def test_on_the_fly_datasets_keep_source_images_disjoint(self):
        discovered_files = discover_upright_image_files(self.data_dir)
        train_files, val_files = split_image_files(discovered_files, seed=11)

        train_dataset = ImageOrientationDataset(train_files, transform=None)
        val_dataset = ImageOrientationDataset(val_files, transform=None)

        expected_rotations = len(config.ROTATIONS)
        self.assertEqual(len(train_dataset), len(train_files) * expected_rotations)
        self.assertEqual(len(val_dataset), len(val_files) * expected_rotations)
        self.assertTrue(set(train_dataset.image_files).isdisjoint(val_dataset.image_files))

    def test_cached_dataset_filters_all_rotations_by_source_id(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        discovered_files = discover_upright_image_files(self.data_dir)
        train_files, val_files = split_image_files(discovered_files, seed=13)

        manifest = []
        for source_path in discovered_files:
            source_id = build_source_id(source_path)
            for label in config.ROTATIONS:
                cached_path = os.path.join(self.cache_dir, f"{source_id}__{label}.png")
                create_image(cached_path, "white")
                manifest.append(
                    {
                        "source_path": os.path.abspath(source_path),
                        "source_id": source_id,
                        "cached_path": os.path.abspath(cached_path),
                        "label": label,
                    }
                )

        with open(get_cache_manifest_path(self.cache_dir), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        train_dataset = ImageOrientationDatasetFromCache(
            self.cache_dir,
            source_ids=[build_source_id(path) for path in train_files],
            transform=None,
        )
        val_dataset = ImageOrientationDatasetFromCache(
            self.cache_dir,
            source_ids=[build_source_id(path) for path in val_files],
            transform=None,
        )

        expected_rotations = len(config.ROTATIONS)
        self.assertEqual(len(train_dataset), len(train_files) * expected_rotations)
        self.assertEqual(len(val_dataset), len(val_files) * expected_rotations)

        train_source_ids = {sample["source_id"] for sample in train_dataset.samples}
        val_source_ids = {sample["source_id"] for sample in val_dataset.samples}
        self.assertTrue(train_source_ids.isdisjoint(val_source_ids))

        train_counts = Counter(sample["source_id"] for sample in train_dataset.samples)
        val_counts = Counter(sample["source_id"] for sample in val_dataset.samples)
        self.assertTrue(all(count == expected_rotations for count in train_counts.values()))
        self.assertTrue(all(count == expected_rotations for count in val_counts.values()))

    def test_build_train_val_datasets_smoke(self):
        args = SimpleNamespace(
            data_dir=self.data_dir,
            seed=17,
            workers=0,
            force_rebuild_cache=False,
        )

        original_use_cache = config.USE_CACHE
        config.USE_CACHE = False
        try:
            train_dataset, val_dataset, train_files, val_files = (
                train_module.build_train_val_datasets(
                    args, {"train": None, "val": None}
                )
            )
        finally:
            config.USE_CACHE = original_use_cache

        expected_rotations = len(config.ROTATIONS)
        self.assertEqual(len(train_dataset), len(train_files) * expected_rotations)
        self.assertEqual(len(val_dataset), len(val_files) * expected_rotations)
        self.assertTrue(set(train_files).isdisjoint(val_files))

    def test_saved_split_state_round_trips_and_overrides_new_seed(self):
        discovered_files = discover_upright_image_files(self.data_dir)
        expected_train_files, expected_val_files = split_image_files(
            discovered_files, seed=19
        )

        train_module.save_split_state(
            self.temp_dir.name,
            split_seed=19,
            train_image_files=expected_train_files,
            val_image_files=expected_val_files,
        )
        saved_split_state = train_module.load_saved_split_state(self.temp_dir.name)

        args = SimpleNamespace(
            data_dir=self.data_dir,
            seed=999,
            workers=0,
            force_rebuild_cache=False,
        )

        original_use_cache = config.USE_CACHE
        config.USE_CACHE = False
        try:
            _, _, train_files, val_files = train_module.build_train_val_datasets(
                args,
                {"train": None, "val": None},
                split_state=saved_split_state,
            )
        finally:
            config.USE_CACHE = original_use_cache

        self.assertEqual(saved_split_state["split_seed"], 19)
        self.assertEqual(train_files, expected_train_files)
        self.assertEqual(val_files, expected_val_files)
        self.assertTrue(set(train_files).isdisjoint(val_files))


if __name__ == "__main__":
    unittest.main()
