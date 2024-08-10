from typing import Optional

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledImageEmbedding:
    embedding: torch.Tensor
    label: str
    source: Path



# Note that the device on which the Tensor lives may be different from the device on which the model is running.
@dataclass
class LabeledImageTensor:
    image: torch.Tensor
    label: str
    source: Path

    @staticmethod
    def collate(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = [item['label'] for item in batch]
        sources = [item['source'] for item in batch]

        return {
            'images': images,
            'labels': labels,
            'sources': sources
        }
