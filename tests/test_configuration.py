import unittest
import torch.nn as nn
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.configuration import Configuration

class ConfigurationTests(unittest.TestCase):
    def test_defaults(self):
        config = Configuration()
        assert config.configuration["model"]["backbone"] == "resnet18"
        assert config.configuration["model"]["pretrained"] == True
        assert config.configuration["model"]["freeze"] == True
        assert config.configuration["model"]["cut"] == -1
        assert config.configuration["model"]["dropout"] == 0.5
        assert config.configuration["model"]["hidden_units"] == 512
        assert config.configuration["train"]["epochs"] == 10
        assert config.configuration["train"]["batch_size"] == 32
        assert config.configuration["train"]["learning_rate"] == 0.001
        assert config.configuration["train"]["momentum"] == 0.9
        assert config.configuration["train"]["weight_decay"] == 0.0001
        assert config.configuration["train"]["device"] == torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")

    def test_load(self):
        config = Configuration(Path("test_configuration.json"))
        assert config.configuration["model"]["backbone"] == "resnet18"
        assert config.configuration["model"]["pretrained"] == True
        assert config.configuration["model"]["freeze"] == True
        assert config.configuration["model"]["cut"] == -1
        assert config.configuration["model"]["dropout"] == 0.5
        assert config.configuration["model"]["hidden_units"] == 512
        assert config.configuration["train"]["epochs"] == 10
        assert config.configuration["train"]["batch_size"] == 32
        assert config.configuration["train"]["learning_rate"] == 0.001
        assert config.configuration["train"]["momentum"] == 0.9
        assert config.configuration["train"]["weight_decay"] == 0.0001
        assert config.configuration["train"]["device"] == torch.device("cpu"), f"Device should be cpu but is {config.configuration['train']['device']}"

    def test_validate_missing_model(self):
        config = Configuration(Path("test_configuration.json"))
        config.configuration.pop("model")
        with self.assertRaises(AssertionError):
            Configuration.validate(config.configuration)
