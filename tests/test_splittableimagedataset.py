import unittest
import torch.nn as nn
import torch
import sys
import os
import platform
from torchvision import transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.SplittableImageDataset import SplittableImageDataset


class SplittableImageDatasetTests(unittest.TestCase):

    def setUp(self):
        if platform.system() == "Darwin":  # macOS
            self.data_dir = os.path.join("/Users", "lobrien", "Dropbox", "shared_src", "mantas", "by_name", "kona")
        elif platform.system() == "Linux":
            self.data_dir = os.path.join("/","mnt", "d", "scratch_data", "mantas", "by_name", "original", "kona")
        else:
            raise Exception(f"Unsupported operating system: {platform.system()}")

    def test_k_over_5(self):
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7)
        # For each label, confirm that there are at least 5 images between train and test
        for label in dataset.class_to_idx.keys():
            train_count = len([x for x in dataset.train_indices if dataset.labels[x] == label])
            test_count = len([x for x in dataset.test_indices if dataset.labels[x] == label])
            self.assertGreaterEqual(train_count, 3)
            self.assertGreaterEqual(test_count, 2)

    def test_find_images_and_labels(self):
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7)
        self.assertEqual(918, len(dataset.image_paths))
        self.assertEqual(918, len(dataset.labels))
        self.assertEqual(56, len(dataset.class_to_idx))
        self.assertEqual('Akari', dataset.labels[0])
        self.assertEqual('Yvet', dataset.labels[-1])

    def test_split_indices(self):
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7)
        self.assertEqual(735, len(dataset.train_indices))
        self.assertEqual(183, len(dataset.test_indices))

    def test_train_test_split(self):
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7)
        self.assertEqual(735, len(dataset.train_indices))
        self.assertEqual(183, len(dataset.test_indices))
        self.assertEqual(15, dataset.train_indices[0])
        self.assertEqual('Akari', dataset.labels[dataset.test_indices[0]])

    def test_transform(self):
        xform = SplittableImageDataset.standard_transform()
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7, transform=xform)
        lit = dataset[0]
        image, label_idx, path = lit.image, lit.label, lit.source
        self.assertEqual(torch.Size([3, 224, 224]), image.size())
        self.assertEqual(0, label_idx)
        self.assertEqual(os.path.join(self.data_dir, 'Akari', 'Akari_20210404_05.jpg'), path)