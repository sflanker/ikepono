from pathlib import Path
from typing import Dict, Any, Tuple
from torch import Model, Dataset
from dataclasses import dataclass

class MockModel(Model):
    def __init__(self, model_configuration: Dict[str, Any]):
        self.model_configuration = model_configuration

class MockDataset(Dataset):
    def __init__(self, images_dir: Path):
        self.images_dir = images_dir

@dataclass
class MockTrainingResults:
    model: Model
    epochs = 10
    precision = 0.9
    recall = 0.9
    f1 = 0.9
    training_start = "2024-01-01T00:00:00"
    training_end = "2024-01-01T00:00:00"


def build_and_train(train_images_dir : Path, val_images_dir : Path, configuration : Dict[str, Any]) -> TrainingResults:
    print(f"Building and training model with images from {train_images_dir}")
    model_configuration, train_configuration = split_configuration(configuration)
    model = training_model(model_configuration)
    train_dataset = image_tensor_dataset(train_images_dir)
    val_dataset = image_tensor_dataset(val_images_dir)
    train_results = train(model, train_dataset, val_dataset, train_configuration)
    return train_results

def split_configuration(configuration: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Mock
    configuration["model"] = { "backbone" : "resnet18", "num_classes": 2, "pretrained": True, "freeze": True, "dropout": 0.5, "hidden_units": 512, "device": "cuda:0"}
    configuration["train"] = { "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0001, "device": "cuda:0"}

    model_configuration = configuration["model"]
    train_configuration = configuration["train"]
    return model_configuration, train_configuration

def training_model(model_configuration: Dict[str, Any]) -> Model:
    print("Building a model")
    return MockModel(model_configuration)

def image_tensor_dataset(images_dir: Path) -> Dataset:
    print(f"Creating a dataset from images in {images_dir}")
    return MockDataset(images_dir)

def train(model: Model, train_dataset: Dataset, val_dataset: Dataset, train_configuration: Dict[str, Any]) -> TrainingResults:
    print("Training the model")
    return MockTrainingResults(model)
