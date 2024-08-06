import unittest
import torch.nn as nn
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.ReidentifyModel import SplittableImageDataset

class SplittableImageDatasetTests(unittest.TestCase):
    data_dir = "/mnt/d/scratch_data/mantas/by_name/kona"

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