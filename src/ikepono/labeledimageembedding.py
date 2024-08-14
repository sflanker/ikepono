from typing import Optional, Union

import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledImageEmbedding:
    embedding: np.ndarray[float]
    label: str
    source: Path
    dataset_index : int



# Note that the device on which the Tensor lives may be different from the device on which the model is running.
@dataclass
class LabeledImageTensor:
    image: torch.Tensor
    label: str
    source: Path

    @staticmethod
    def collate(labeledImageTensors : list["LabeledImageTensor"]) -> dict[str, Union[torch.Tensor, np.ndarray]]:
        images = torch.stack([item.image for item in labeledImageTensors])
        labels = np.array([item.label for item in labeledImageTensors])
        sources = np.array([item.source for item in labeledImageTensors])

        return {
            'images': images,
            'labels': labels,
            'sources': sources
        }
