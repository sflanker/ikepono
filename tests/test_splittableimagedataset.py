import unittest
import torch.nn as nn
import torch
import sys
import os
import platform
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.SplittableImageDataset import SplittableImageDataset


class SplittableImageDatasetTests(unittest.TestCase):

    def setUp(self):
        if platform.system() == "Darwin":  # macOS
            self.data_dir = os.path.join("/Users", "lobrien", "Dropbox", "shared_src", "mantas", "by_name", "kona")
        elif platform.system() == "Windows":
            self.data_dir = os.path.join("D:", "scratch_data", "mantas", "by_name", "kona")
        else:
            raise RuntimeError(f"Unsupported operating system: {platform.system()}")

    def test_k_over_5(self):
        assert False
    def test_find_images_and_labels(self):
        dataset = SplittableImageDataset(root_dir=self.data_dir, k=7)
        assert len(dataset.image_paths) == 4
        assert len(dataset.labels) == 4
        assert dataset.class_to_idx == {'class1': 0, 'class2': 1}

    def test_split_indices(self):
        dataset = SplittableImageDataset(root_dir="data", k=7)
        assert len(dataset.train_indices) == 3
        assert len(dataset.test_indices) == 1

    def test_train_test_split(self):
        dataset = SplittableImageDataset(root_dir="data", k=7)
        assert len(dataset.train_indices) == 3
        assert len(dataset.test_indices) == 1
        assert dataset.labels[dataset.train_indices[0]] == 'class1'
        assert dataset.labels[dataset.test_indices[0]] == 'class2'