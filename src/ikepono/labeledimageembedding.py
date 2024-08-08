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