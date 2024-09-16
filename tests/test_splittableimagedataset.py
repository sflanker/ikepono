import os
import platform
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.splittableimagedataset import SplittableImageDataset


class SplittableImageDatasetTests(unittest.TestCase):

    @classmethod
    def simple_dataset(cls):
        data_dir = Path("/mnt/d/scratch_data/mantas/by_name/original_2023/kona")
        akara_dir = os.path.join(data_dir, 'Akari')
        akari_image_files = [os.path.join(akara_dir, f) for f in os.listdir(akara_dir) if f.endswith('.jpg')][:10]
        yvet_dir = os.path.join(data_dir, 'Vallaray')
        yvet_image_files = [os.path.join(yvet_dir, f) for f in os.listdir(yvet_dir) if f.endswith('.jpg')][:10]
        paths = akari_image_files + yvet_image_files
        akari_labels = ['Akari'] * len(akari_image_files)
        yvet_labels = ['Vallaray'] * len(yvet_image_files)
        labels = akari_labels + yvet_labels
        assert len(paths) == len(labels)
        assert len(paths) == 20
        dataset = SplittableImageDataset(paths=paths, labels=labels, k=4)
        dataset.train_indices = [np.int64(x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]]
        dataset.test_indices =  [np.int64(x) for x in [8, 9, 18, 19]]
        return dataset

    def setUp(self):
        if platform.system() == "Darwin":  # macOS
            self.data_dir = os.path.join("/Users", "lobrien", "Dropbox", "shared_src", "mantas", "by_name", "kona")
        elif platform.system() == "Linux":
            self.data_dir = os.path.join("/","mnt", "d", "scratch_data", "mantas", "by_name", "original", "kona")
        else:
            raise Exception(f"Unsupported operating system: {platform.system()}")

    def test_k_over_5(self):
        k  = 7
        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=k)
        # For each label, confirm that there are at least 5 images between train and test
        for label in dataset.label_to_idx.keys():
            train_count = len([x for x in dataset.train_indices if dataset.labels[x] == label])
            test_count = len([x for x in dataset.test_indices if dataset.labels[x] == label])
            if train_count < k:
                print(f"Expected at least {k} train images for {label}, got train|test {train_count} | {test_count}")
            self.assertGreaterEqual(train_count, k, f"Expected at least {k} train images for {label}, got train|test {train_count} | {test_count}")
            self.assertGreaterEqual(test_count, 1, f"Expected at least 1 test images for {label}, got train|test {train_count} | {test_count}")

    def test_find_images_and_labels(self):
        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=7)
        self.assertEqual(814, len(dataset.image_paths))
        self.assertEqual(42, len(dataset.label_to_idx))
        self.assertEqual('Akari', dataset.labels[0])
        self.assertEqual('Yvet', dataset.labels[-1])

    def test_explicit_path_constructor(self):
        dataset = SplittableImageDatasetTests.simple_dataset()
        self.assertEqual(20, len(dataset.image_paths))
        self.assertEqual(20, len(dataset.labels))
        self.assertEqual(2, len(dataset.label_to_idx))
        self.assertEqual('Akari', dataset.labels[0])
        self.assertEqual('Vallaray', dataset.labels[10])

    def test_split_indices(self):
        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=7)
        self.assertEqual(601, len(dataset.train_indices))
        self.assertEqual(213, len(dataset.test_indices))

    def test_train_test_split(self):
        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=7)
        self.assertEqual(601, len(dataset.train_indices))
        self.assertEqual(213, len(dataset.test_indices))
        self.assertEqual(0, dataset.train_indices[0])
        self.assertEqual('Akari', dataset.labels[dataset.test_indices[0]])

    def test_transform(self):
        xform = SplittableImageDataset.standard_transform()
        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=7, transform=xform)
        lit = dataset[0]
        image, label_idx, path = lit.image, lit.label_idx, lit.source
        self.assertEqual(torch.Size([3, 224, 224]), image.size())
        self.assertEqual(0, label_idx)
        self.assertEqual(os.path.join(self.data_dir, 'Akari', 'Akari_20140330_01.jpg'), path)

if __name__ == '__main__':
    unittest.main()