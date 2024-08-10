from pathlib import Path
from typing import Optional
import torch
import json

class Configuration:
    def __init__(self, config_file : Optional[Path] = None):
        self.configuration = {}
        if config_file is not None:
            self.configuration = self.load(config_file)
        else:
            self.configuration = Configuration.defaults()
        self.configuration["train"]["dataset_device"] = torch.device(self.configuration["train"]["dataset_device"])
        self.configuration["train"]["model_device"] = torch.device(self.configuration["train"]["model_device"])
        Configuration.validate(self.configuration)

    @staticmethod
    def validate(configuration):
        assert "model" in configuration, "model configuration is missing"
        assert "train" in configuration, "train configuration is missing"
        assert "epochs" in configuration["train"], "epochs is missing from train configuration"
        assert "batch_size" in configuration["train"], "batch_size is missing from train configuration"
        assert "learning_rate" in configuration["train"], "learning_rate is missing from train configuration"
        assert "momentum" in configuration["train"], "momentum is missing from train configuration"
        assert "weight_decay" in configuration["train"], "weight_decay is missing from train configuration"
        assert "backbone" in configuration["model"], "backbone is missing from model configuration"
        assert "pretrained" in configuration["model"], "pretrained is missing from model configuration"
        assert "freeze" in configuration["model"], "freeze is missing from model configuration"
        assert "cut" in configuration["model"], "cut is missing from model configuration"
        assert "dropout" in configuration["model"], "dropout is missing from model configuration"
        assert "hidden_units" in configuration["model"], "hidden_units is missing from model configuration"
        assert "output_vector_size" in configuration["model"], "output_vector_size is missing from model configuration"
        assert "dataset_device" in configuration["model"], "dataset device is missing from model configuration"
        assert "model_device" in configuration["model"], "model device is missing from model configuration"

        assert 2 == len(configuration), "Configuration should have exactly two keys: model and train"
        assert 7 == len(configuration["train"]), "Train configuration should have exactly 6 keys: epochs, batch_size, learning_rate, momentum, weight_decay, dataset_device, model_device"
        assert 10 == len(configuration["model"]), "Model configuration should have exactly 10 keys: backbone, pretrained, freeze, cut, dropout, backbone_output_dim, hidden_units, output_vector_size, dataset_device, model_device"

    @staticmethod
    def defaults():
        configuration = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
        configuration["model"] = {"backbone": "resnet18", "pretrained": True, "freeze": True, "cut": -1, "dropout": 0.5,
                              "backbone_output_dim": 512, "hidden_units": 512, "output_vector_size": 128, "dataset_device": torch.device("cpu"),
                                  "model_device" : device}

        configuration["train"] = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001, "momentum": 0.9,
                              "weight_decay": 0.0001, "dataset_device": torch.device("cpu"),
                                  "model_device" : device}
        return configuration

    def load(self, config_file : Path):
        with open(config_file, 'r') as f:
            configuration = json.load(f)
        return configuration

    def train_configuration(self):
        return self.configuration["train"]

    def model_configuration(self):
        return self.configuration["model"]
