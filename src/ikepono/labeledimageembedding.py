import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabeledImageEmbedding:
    embedding: np.ndarray[float]
    label: str
    source: Path
    dataset_index : int

