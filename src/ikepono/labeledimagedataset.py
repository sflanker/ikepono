import torch
from torch.utils.data import Dataset
from torchvision import transforms as xforms

import numpy as np
import os
from PIL import Image
from ikepono.indexedimagetensor import IndexedImageTensor
from pathlib import Path

class LabeledImageDataset(Dataset):

    @classmethod
    def from_directory(cls, root_dir, transform=None, device = torch.device('cpu'), k:int = 5) -> "LabeledImageDataset":
        if transform is None:
            transform = LabeledImageDataset.standard_transform_for_training()
        image_paths = []
        for root, _, files in os.walk(root_dir):
            # Only add direactory if it has at least k images
            if len(files) >= k:
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        full_path = Path(os.path.join(root, file))
                        image_paths.append(full_path)
        return cls(image_paths, transform, device)

    @classmethod
    def reconcile(cls, train_ds: "LabeledImageDataset", validation_ds: "LabeledImageDataset") -> None:
        # If a label does not occur in both datasets, remove it from the larger dataset
        train_labels = set(train_ds.labels)
        validation_labels = set(validation_ds.labels)
        labels_to_remove = train_labels.symmetric_difference(validation_labels)
        if len(labels_to_remove) > 0:
            for label in labels_to_remove:
                if label in train_labels:
                    train_ds.labels = train_ds.labels[train_ds.labels != label]
                if label in validation_labels:
                    validation_ds.labels = validation_ds.labels[validation_ds.labels != label]

    def __init__(self, paths, transform: xforms.Compose, train=True, device = torch.device('cpu')):
        assert len(paths) > 0, "No images found in paths"
        assert transform is not None, "transform cannot be None"
        assert device is not None, "device cannot be None"

        self.image_paths = paths
        self.transform = transform
        self.labels = np.unique([p.parent.name for p in paths])
        self.device = device
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.source_to_label_idx = {p: self.label_to_idx[p.parent.name] for p in paths}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}


    def __len__(self) -> int:
            return len(self.image_paths)

    def __getitem__(self, idx : int) ->IndexedImageTensor:
            img_path = self.image_paths[idx]

            initial_image = Image.open(img_path)
            pil_image = initial_image.convert('RGB')
            try:
                label_idx = self.source_to_label_idx[img_path]

                if self.transform:
                    tensor_image = self.transform(pil_image)
                else:
                    # Minimum xform is to tensor
                    transform = xforms.Compose([xforms.ToTensor()])
                    tensor_image = transform(pil_image)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                raise
            finally:
                initial_image.close()

            tensor_image = tensor_image.to(self.device)
            return IndexedImageTensor(image=tensor_image, label_idx=label_idx, source=img_path)

    @staticmethod
    def standard_transform_for_inference() -> xforms.Compose:
        return xforms.Compose([
            xforms.Resize((224,224)),
            xforms.ToTensor(),
            xforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    @staticmethod
    def standard_transform_for_training() -> xforms.Compose:
        return xforms.Compose([
            xforms.ToTensor(),
            xforms.RandomHorizontalFlip(p=0.5),
            xforms.RandomCrop(size=(224, 224), padding=4, pad_if_needed=True),
            xforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            xforms.RandomRotation(degrees=10),
            xforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            xforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            xforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            xforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))  # Gaussian noise
        ])