import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledImageEmbedding:
    embedding: torch.Tensor
    label: str
    source: Path

@dataclass
class LabeledImageTensor:
    image: torch.Tensor
    label: str
    source: Path