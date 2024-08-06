import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels, self.class_to_idx = self._find_images_and_labels()

    def _find_images_and_labels(self):
        image_paths = []
        labels = []
        class_to_idx = {}

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)

                    # Use parent directory name as label
                    label = os.path.basename(os.path.dirname(full_path))
                    labels.append(label)

                    # Build class_to_idx dictionary
                    if label not in class_to_idx:
                        class_to_idx[label] = len(class_to_idx)

        return image_paths, labels, class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Return the image, label, and path
        return image, label, idx, img_path

