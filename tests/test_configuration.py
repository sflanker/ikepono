import os
import sys
import unittest
from pathlib import Path

import torch

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
        assert config.configuration["model"]["backbone_output_dim"] == 512
        assert config.configuration["model"]["hidden_units"] == 512
        assert config.configuration["train"]["epochs"] == 10
        assert config.configuration["train"]["learning_rate"] == 0.001
        assert config.configuration["train"]["dataset_device"] == torch.device("cpu")
        assert config.configuration["train"]["model_device"] == torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")

    def test_load(self):
        config = Configuration(Path("test_configuration.json"))
        assert config.configuration["model"]["backbone"] == "resnet18"
        assert config.configuration["model"]["pretrained"] == True
        assert config.configuration["model"]["freeze"] == True
        assert config.configuration["model"]["cut"] == -1
        assert config.configuration["model"]["dropout"] == 0.5
        assert config.configuration["model"]["hidden_units"] == 512
        assert config.configuration["train"]["epochs"] == 2
        assert config.configuration["train"]["learning_rate"] == 0.001
        assert config.configuration["train"]["dataset_device"] == torch.device("cpu"), f"Device should be cpu but is {config.configuration['train']['dataset_device']}"
        assert config.configuration["train"]["model_device"] == torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu"), f"Device should be cpu but is {config.configuration['train']['device']}"

    def test_validate_missing_model(self):
        config = Configuration(Path("test_configuration.json"))
        config.configuration.pop("model")
        with self.assertRaises(AssertionError):
            Configuration.validate(config.configuration)


if __name__ == "__main__":
    unittest.main()
