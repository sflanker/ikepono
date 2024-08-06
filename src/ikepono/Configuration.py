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
        assert "device" in configuration["train"], "device is missing from train configuration"
        assert "backbone" in configuration["model"], "backbone is missing from model configuration"
        assert "pretrained" in configuration["model"], "pretrained is missing from model configuration"
        assert "freeze" in configuration["model"], "freeze is missing from model configuration"
        assert "cut" in configuration["model"], "cut is missing from model configuration"
        assert "dropout" in configuration["model"], "dropout is missing from model configuration"
        assert "hidden_units" in configuration["model"], "hidden_units is missing from model configuration"
        assert "device" in configuration["model"], "device is missing from model configuration"

        assert 2 == len(configuration), "Configuration should have exactly two keys: model and train"
        assert 6 == len(configuration["train"]), "Train configuration should have exactly 6 keys: epochs, batch_size, learning_rate, momentum, weight_decay, device"
        assert 7 == len(configuration["model"]), "Model configuration should have exactly 7 keys: backbone, pretrained, freeze, cut, dropout, hidden_units, device"

    @staticmethod
    def defaults():
        configuration = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
        configuration["model"] = {"backbone": "resnet18", "pretrained": True, "freeze": True, "cut": -1, "dropout": 0.5,
                              "hidden_units": 512, "device": device}

        configuration["train"] = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001, "momentum": 0.9,
                              "weight_decay": 0.0001, "device": device}
        return configuration

    def load(self, config_file : Path):
        with open(config_file, 'r') as f:
            configuration = json.load(f)
        configuration["train"]["device"] = torch.device(configuration["train"]["device"])
        return configuration
