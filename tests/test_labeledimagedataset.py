import os
import torch
import unittest

from ikepono.configuration import Configuration
from ikepono.indexedimagetensor import IndexedImageTensor
from ikepono.labeledimagedataset import LabeledImageDataset


class LabeledImageDatasetTests(unittest.TestCase):
    def test_fromdir(self):
        config = Configuration("test_configuration.json")
        root_src_dir = config.configuration["train"]["data_path"]
        train_dir = os.path.join(root_src_dir, "train")
        validation_dir = os.path.join(root_src_dir, "valid")
        device = torch.device(config.configuration["train"]["dataset_device"])
        n_triplets = config.configuration["train"]["n_triplets"]

        train_ds = LabeledImageDataset.from_directory(train_dir, transform=None, device = device)
        validation_ds = LabeledImageDataset.from_directory(validation_dir, transform=None, device = device)
        LabeledImageDataset.reconcile(train_ds, validation_ds)
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(validation_ds)
        train_labels = train_ds.labels
        validation_labels = validation_ds.labels
        self.assertIsNotNone(train_labels)
        self.assertIsNotNone(validation_labels)
        self.assertEqual(len(train_labels), len(validation_labels))
        self.assertGreater(len(train_labels), 0)

        loader = torch.utils.data.DataLoader(train_ds, batch_size=n_triplets, shuffle=True, collate_fn=IndexedImageTensor.collate)
        self.assertIsNotNone(loader)

    def test_reconcile(self):
        config = Configuration("test_configuration.json")
        root_src_dir = config.configuration["train"]["data_path"]
        train_dir = os.path.join(root_src_dir, "train")
        validation_dir = os.path.join(root_src_dir, "valid")
        device = torch.device(config.configuration["train"]["dataset_device"])

        train_ds = LabeledImageDataset.from_directory(train_dir, transform=None, device = device)
        validation_ds = LabeledImageDataset.from_directory(validation_dir, transform=None, device = device)
        LabeledImageDataset.reconcile(train_ds, validation_ds)
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(validation_ds)
        train_labels = train_ds.labels
        validation_labels = validation_ds.labels
        self.assertIsNotNone(train_labels)
        self.assertIsNotNone(validation_labels)
        self.assertEqual(len(train_labels), len(validation_labels))
        self.assertGreater(len(train_labels), 0)


if __name__ == "__main__":
    unittest.main()
