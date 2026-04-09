import json
import os
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import config
import src.caching as caching_module
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


def create_corner_marked_image(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.new("RGB", (3, 2), "white")
    pixels = image.load()
    pixels[0, 0] = (255, 0, 0)
    pixels[2, 0] = (0, 255, 0)
    pixels[0, 1] = (0, 0, 255)
    pixels[2, 1] = (255, 255, 0)
    image.save(path)


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

    def test_on_the_fly_dataset_applies_expected_right_angle_rotations(self):
        image_path = os.path.join(self.data_dir, "geometry", "corner_markers.png")
        create_corner_marked_image(image_path)

        dataset = ImageOrientationDataset(
            [image_path], transform=lambda image: image.copy()
        )

        expected_images = {
            0: {
                "size": (3, 2),
                "corners": {
                    (0, 0): (255, 0, 0),
                    (2, 0): (0, 255, 0),
                    (0, 1): (0, 0, 255),
                    (2, 1): (255, 255, 0),
                },
            },
            1: {
                "size": (2, 3),
                "corners": {
                    (0, 0): (0, 255, 0),
                    (1, 0): (255, 255, 0),
                    (0, 2): (255, 0, 0),
                    (1, 2): (0, 0, 255),
                },
            },
            2: {
                "size": (3, 2),
                "corners": {
                    (0, 0): (255, 255, 0),
                    (2, 0): (0, 0, 255),
                    (0, 1): (0, 255, 0),
                    (2, 1): (255, 0, 0),
                },
            },
            3: {
                "size": (2, 3),
                "corners": {
                    (0, 0): (0, 0, 255),
                    (1, 0): (255, 0, 0),
                    (0, 2): (255, 255, 0),
                    (1, 2): (0, 255, 0),
                },
            },
        }

        for label, expectation in expected_images.items():
            rotated_image, returned_label = dataset[label]
            self.assertEqual(returned_label.item(), label)
            self.assertEqual(rotated_image.size, expectation["size"])
            for coordinates, expected_color in expectation["corners"].items():
                self.assertEqual(rotated_image.getpixel(coordinates), expected_color)

    def test_on_the_fly_dataset_rejects_unsupported_rotation_config(self):
        image_path = os.path.join(self.data_dir, "geometry", "plain.png")
        create_image(image_path, "white")

        original_rotations = config.ROTATIONS
        self.addCleanup(setattr, config, "ROTATIONS", original_rotations)
        config.ROTATIONS = {0: 0, 1: 45}

        with self.assertRaisesRegex(
            ValueError, "Unsupported rotation angles in config.ROTATIONS"
        ):
            ImageOrientationDataset([image_path], transform=None)

    def test_on_the_fly_dataset_does_not_mask_loading_failures(self):
        missing_path = os.path.join(self.data_dir, "geometry", "missing.png")
        dataset = ImageOrientationDataset([missing_path], transform=None)

        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_cached_dataset_does_not_mask_loading_failures(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        create_image(os.path.join(self.cache_dir, "present.png"), "white")
        sample = {
            "cached_path": os.path.join(self.cache_dir, "missing.png"),
            "label": 0,
        }
        dataset = ImageOrientationDatasetFromCache(
            self.cache_dir,
            samples=[sample],
            transform=None,
        )

        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_split_helper_rejects_invalid_train_ratio(self):
        discovered_files = discover_upright_image_files(self.data_dir)

        for train_ratio in (0, 1, 1.2, -0.1):
            with self.assertRaises(ValueError):
                split_image_files(discovered_files, train_ratio=train_ratio)

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

    def test_cached_dataset_warns_when_manifest_is_missing_requested_source_ids(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        discovered_files = discover_upright_image_files(self.data_dir)
        requested_files = discovered_files[:3]
        cached_files = requested_files[:2]

        manifest = []
        for source_path in cached_files:
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

        requested_source_ids = [build_source_id(path) for path in requested_files]
        with self.assertLogs(level="WARNING") as captured_logs:
            dataset = ImageOrientationDatasetFromCache(
                self.cache_dir,
                source_ids=requested_source_ids,
                transform=None,
            )

        self.assertEqual(len(dataset), len(cached_files) * len(config.ROTATIONS))
        self.assertTrue(
            any(
                "Cache manifest is missing 1 requested source IDs" in line
                for line in captured_logs.output
            )
        )

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

    def test_build_train_val_datasets_threads_cli_cache_args(self):
        args = SimpleNamespace(
            data_dir=self.data_dir,
            seed=17,
            workers=3,
            force_rebuild_cache=True,
        )

        original_use_cache = config.USE_CACHE
        config.USE_CACHE = True
        try:
            with mock.patch.object(train_module, "cache_dataset") as cache_dataset_mock, mock.patch.object(
                train_module,
                "ImageOrientationDatasetFromCache",
                side_effect=["train-cache", "val-cache"],
            ):
                train_dataset, val_dataset, _, _ = train_module.build_train_val_datasets(
                    args, {"train": None, "val": None}
                )
        finally:
            config.USE_CACHE = original_use_cache

        cache_dataset_mock.assert_called_once_with(
            upright_dir=self.data_dir,
            num_workers=3,
            force_rebuild=True,
        )
        self.assertEqual(train_dataset, "train-cache")
        self.assertEqual(val_dataset, "val-cache")

    def test_build_train_val_datasets_preserves_zero_workers_for_cache(self):
        args = SimpleNamespace(
            data_dir=self.data_dir,
            seed=17,
            workers=0,
            force_rebuild_cache=False,
        )

        original_use_cache = config.USE_CACHE
        config.USE_CACHE = True
        try:
            with mock.patch.object(train_module, "cache_dataset") as cache_dataset_mock, mock.patch.object(
                train_module,
                "ImageOrientationDatasetFromCache",
                side_effect=["train-cache", "val-cache"],
            ):
                train_module.build_train_val_datasets(args, {"train": None, "val": None})
        finally:
            config.USE_CACHE = original_use_cache

        cache_dataset_mock.assert_called_once_with(
            upright_dir=self.data_dir,
            num_workers=0,
            force_rebuild=False,
        )

    def test_cache_dataset_honors_upright_dir_and_zero_workers(self):
        cache_dir = os.path.join(self.temp_dir.name, "cache_out")
        original_cache_dir = config.CACHE_DIR
        original_data_dir = config.DATA_DIR
        config.CACHE_DIR = cache_dir
        config.DATA_DIR = os.path.join(self.temp_dir.name, "missing_data_dir")
        self.addCleanup(setattr, config, "CACHE_DIR", original_cache_dir)
        self.addCleanup(setattr, config, "DATA_DIR", original_data_dir)

        with mock.patch.object(
            caching_module, "Pool", side_effect=AssertionError("Pool should not be used")
        ):
            caching_module.cache_dataset(
                upright_dir=self.data_dir,
                num_workers=0,
                force_rebuild=True,
            )

        manifest_path = get_cache_manifest_path(cache_dir)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(len(manifest), len(self.image_paths) * len(config.ROTATIONS))
        cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".png")]
        self.assertEqual(len(cached_files), len(manifest))

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

    def test_load_saved_split_state_rejects_non_object_json(self):
        split_state_path = train_module.get_split_state_path(self.temp_dir.name)
        with open(split_state_path, "w", encoding="utf-8") as handle:
            json.dump(["not", "an", "object"], handle)

        with self.assertRaises(ValueError):
            train_module.load_saved_split_state(self.temp_dir.name)


if __name__ == "__main__":
    unittest.main()
