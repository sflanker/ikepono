import torch

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# Note that the device on which the Tensor lives may be different from the device on which the model is running.
@dataclass
class IndexedImageTensor:
    image: torch.Tensor
    # This index is the index of the label in the dataset's list of labels
    # i.e., label_str = train_dataset.labels[label_idx]
    label_idx: int
    source: Path

    @staticmethod
    def collate(indexed_image_tensors: list["IndexedImageTensor"]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item.image for item in indexed_image_tensors])
        label_indexes = torch.Tensor([int(item.label_idx) for item in indexed_image_tensors]).type(torch.int64)
        sources = np.array([item.source for item in indexed_image_tensors])
        rv = {
            "images": images,
            "label_indexes": label_indexes,
            "sources": sources
        }

        return rv
