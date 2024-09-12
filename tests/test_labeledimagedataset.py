import os
import torch
import unittest

from ikepono.configuration import Configuration
from ikepono.labeledimagedataset import LabeledImageDataset
from ikepono.labeledimageembedding import LabeledImageTensor


class LabeledImageDatasetTests(unittest.TestCase):
    def test_fromdir(self):
        config = Configuration("test_configuration.json")
        root_src_dir = config.train_configuration()["data_path"]
        train_dir = os.path.join(root_src_dir, "train")
        validation_dir = os.path.join(root_src_dir, "valid")
        device = torch.device(config.train_configuration()["dataset_device"])
        n_triplets = config.train_configuration()["n_triplets"]

        train_ds = LabeledImageDataset.from_directory(train_dir, transform=None, device = device)
        validation_ds = LabeledImageDataset.from_directory(validation_dir, transform=None, device = device)
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(validation_ds)
        train_labels = train_ds.labels
        validation_labels = validation_ds.labels
        self.assertIsNotNone(train_labels)
        self.assertIsNotNone(validation_labels)
        self.assertEqual(len(train_labels), len(validation_labels))
        self.assertGreater(len(train_labels), 0)

        loader = torch.utils.data.DataLoader(train_ds, batch_size=n_triplets, shuffle=True, collate_fn=LabeledImageTensor.collate)
        self.assertIsNotNone(loader)



if __name__ == "__main__":
    unittest.main()
