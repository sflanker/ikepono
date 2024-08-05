from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Iterable, Optional
from torchvision.tv_tensors import Image # This is subclass of Tensor
import torch

@dataclass
class Identification:
    name: str
    confidence: float
    cropped_image: Image
def get_inbox_images(inbox: Path) -> Iterable[Path]:
    files = list(inbox.iterdir())
    return map(crop_all, files)

def crop_all(input_path: Path) -> Iterable[Tuple[Path, Image]]:
    return map(detect, input_path)

def save_all_crops(crops: Iterable[Tuple[Path,Image]], output_dir: Path) -> List[Path]:
    path = crops[0]
    image_tensor = crops[1]
    i = 0
    for crop in crops:
        crop_name = f"{path.stem}_crop_{i}.png"
        i += 1
        save_crop(output_dir, crop_name, image_tensor)

def save_crop(output_dir: Path, crop_name : str, image_t : Image) -> Path:
    path = Path(output_dir / crop_name)
    print(f"Saving crop {path}")
    return path

def detect(image_path: Path) -> Iterable[Image]:
    print(f"Detecting mantas in {image_path}")
    for _ in range(3):
        mock_image = torch.rand(3, 224, 224)
        yield mock_image
def reidentify(cropped_image : Image) -> List[Identification]:
    identifications = []
    for i in range(3):
        identifications.append(Identification(name=f"manta_{i}", confidence=0.9, cropped_image=cropped_image))
    return identifications

def present_identifications(identifications: List[Identification], output_dir: Path) -> Optional[Identification]:
    print(f"Selecting best identification from identifications")
    identification = max(identifications, key=lambda x: x.confidence)
    return identification

def identify_inbox(inbox: Path, output_dir: Path) -> Optional[Identification]:
    inbox_images = get_inbox_images(inbox)
    crops = crop_all(inbox_images)
    proposed_identifications = reidentify(crops)
    identification = present_identifications(proposed_identifications, output_dir)
    return identification
