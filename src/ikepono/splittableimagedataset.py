import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
from PIL import Image
from collections import defaultdict
from ikepono.indexedimagetensor import IndexedImageTensor
from deprecated import deprecated
from pathlib import Path
from sklearn.model_selection import train_test_split


# TODO: Delete this if datasets from train/ valid/ directories are solely to be used
@deprecated(reason="Use LabeledImageDataset instead. Better for analysis.")
class SplittableImageDataset(Dataset):
    @classmethod
    def from_directory(cls, root_dir, transform=None, train=True, test_size=0.2, random_state=42, k=5, device = torch.device('cpu')) -> "SplittableImageDataset":
        image_paths = []
        labels = []
        class_counts = defaultdict(int)

        # First pass: count images per class
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    label = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    class_counts[label] += 1

        # Second pass: keep only classes with at least k members
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    full_path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(full_path))

                    if class_counts[label] >= (k + k * test_size):
                        image_paths.append(full_path)
                        labels.append(label)

        return cls(image_paths, labels, transform, train, test_size, random_state, k, device)


    def __init__(self, paths, labels, transform=None, train=True, test_size=0.2, random_state=42, k=5, device = torch.device('cpu')):
        self.root_dir = None
        if transform is None:
            transform = SplittableImageDataset.standard_transform()
        self.transform = transform
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.k = k
        self.image_paths = paths
        self.labels = labels
        self.device = device
        self.label_to_idx = {label: idx for idx, label in enumerate(np.unique(labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.train_indices, self.test_indices = self._split_indices()
        self.source_to_label = {}
        for p in paths:
            path = Path(p)
            self.source_to_label[path] = path.parent.name

    def _split_indices(self) -> tuple[list[int], list[int]]:
        indices = np.arange(len(self.image_paths))
        labels = np.array(self.labels)
        assert indices.shape == labels.shape, "Indices and labels must have the same shape. Labels need to be repeated for each image."

        train_indices, test_indices = [], []

        for class_label in np.unique(labels):
            class_indices = indices[labels == class_label]
            n_samples = len(class_indices)

            if n_samples < self.k:  # Minimum 3 for train and 2 for test
                raise ValueError(f"Class {class_label} has fewer than 5 samples.")

            # Add the first 3 samples of class_label to train
            train_indices.extend(class_indices[:3])

            # Add the next 2 samples of class_label to test
            test_indices.extend(class_indices[3:5])

            n_test = int((n_samples-5) * self.test_size)
            n_train = (n_samples - 5) - n_test

            # Add the rest of the samples to train and test
            if n_test > 0 and n_train > 0:
                train_additionals, test_additionals = train_test_split(
                    class_indices[5:], test_size=self.test_size, train_size=(1 - self.test_size),
                    random_state=self.random_state
                )
                train_indices.extend(train_additionals)
                test_indices.extend(test_additionals)
            elif n_train > 0:
                train_indices.extend(class_indices[5:])

        return train_indices, test_indices

    def __len__(self) -> int:
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx : int) ->IndexedImageTensor:
        img_path = self.image_paths[idx]

        initial_image = Image.open(img_path)
        pil_image = initial_image.convert('RGB')
        try:
            label = self.source_to_label[Path(img_path)]
            label_idx = self.label_to_idx[label]

            if self.transform:
                tensor_image = self.transform(pil_image)
            else:
                # Minimum xform is to tensor
                transform = transforms.Compose([transforms.ToTensor()])
                tensor_image = transform(pil_image)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            raise
        finally:
            initial_image.close()
        # Move it on to configuration["dataset_device"]
        tensor_image = tensor_image.to(self.device)

        return IndexedImageTensor(image=tensor_image, label_idx=label_idx, source=img_path)

    @staticmethod
    def standard_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
