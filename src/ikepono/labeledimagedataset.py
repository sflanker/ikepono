import numpy as np
import os
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as xforms

from ikepono.labeledimageembedding import LabeledImageTensor


# TODO: Delete this if datasets from train/ valid/ directories are solely to be used
class LabeledImageDataset(Dataset):

    @classmethod
    def from_directory(cls, root_dir, transform=None, device = torch.device('cpu')) -> "LabeledImageDataset":
        if transform is None:
            transform = LabeledImageDataset.standard_transform()
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    full_path = Path(os.path.join(root, file))
                    image_paths.append(full_path)
        return cls(image_paths, transform, device)

    def __init__(self, paths, transform: xforms.Compose, train=True, device = torch.device('cpu')):
        assert len(paths) > 0, "No images found in paths"
        assert transform is not None, "transform cannot be None"
        assert device is not None, "device cannot be None"

        self.image_paths = paths
        self.transform = transform
        self.labels = np.unique([p.parent.name for p in paths])
        self.device = device
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx : int) ->LabeledImageTensor:
        img_path = self.image_paths[idx]

        initial_image = Image.open(img_path)
        pil_image = initial_image.convert('RGB')
        try:
            label = self.labels[idx]

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
        return LabeledImageTensor(image=tensor_image, label=self.label_to_idx[label], source=img_path)

    @staticmethod
    def standard_transform() -> xforms.Compose:
        return xforms.Compose([
            xforms.Resize((224,224)),
            xforms.ToTensor(),
            xforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
