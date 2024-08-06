import unittest
import torch.nn as nn
import torch
import sys
import os
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.ImageDataset import ImageDataset

class DatasetTests(unittest.TestCase):
    data_dir = Path("/mnt/d/scratch_data/mantas/by_name/original/kona/")
    def test_training_directories(self):
        assert os.path.exists(self.data_dir)

    def test_dataset(self):
        tforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = ImageDataset(self.data_dir, transform=tforms)
        assert len(dataset) > 0
        for img, label, index, path in dataset:
            assert img.shape == (3, 224, 224)
            assert label == "'Ele'ele-Ka'ohe"
            assert path.name == "'Ele'ele-Ka'ohe_20070804_01.jpg"
            assert index == 0
            break


if __name__ == '__main__':
    unittest.main()