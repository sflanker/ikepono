import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from sklearn.model_selection import train_test_split


class SplittableImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test_size=0.2, random_state=42, k=5):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.k = k
        self.image_paths, self.labels, self.class_to_idx = self._find_images_and_labels()
        self.train_indices, self.test_indices = self._split_indices()

    def _find_images_and_labels(self):
        image_paths = []
        labels = []
        class_to_idx = {}
        class_counts = defaultdict(int)

        # First pass: count images per class
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    label = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    class_counts[label] += 1

        # Second pass: keep only classes with at least k members
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    full_path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(full_path))

                    if class_counts[label] >= self.k:
                        image_paths.append(full_path)
                        labels.append(label)

                        if label not in class_to_idx:
                            class_to_idx[label] = len(class_to_idx)

        return image_paths, labels, class_to_idx

    def _split_indices(self):
        indices = np.arange(len(self.image_paths))
        labels = np.array(self.labels)

        train_indices, test_indices = [], []

        for class_label in np.unique(labels):
            class_indices = indices[labels == class_label]
            n_samples = len(class_indices)

            if n_samples < 5:  # Minimum 3 for train and 2 for test
                raise ValueError(f"Class {class_label} has fewer than 5 samples.")

            n_test = max(2, int(n_samples * self.test_size))
            n_train = n_samples - n_test

            if n_train < 3:
                n_train = 3
                n_test = n_samples - n_train

            class_train, class_test = train_test_split(
                class_indices, test_size=n_test, train_size=n_train,
                random_state=self.random_state
            )

            train_indices.extend(class_train)
            test_indices.extend(class_test)

        return train_indices, test_indices

    def __len__(self):
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx):
        if self.train:
            idx = self.train_indices[idx]
        else:
            idx = self.test_indices[idx]

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label], img_path

    @staticmethod
    def standard_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
