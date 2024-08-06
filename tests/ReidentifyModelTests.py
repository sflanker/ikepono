import unittest
import torch.nn as nn
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.ReidentifyModel import _init_weights, ReidentifyModel


class ReidentifyModelTests(unittest.TestCase):
    def test_initialize_weights_linear(self):
        # Create a linear layer with input size 100 and output size 100
        linear_layer = nn.Linear(100, 100)

        # Initialize the weights and bias to zeros
        nn.init.zeros_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)

        _init_weights(linear_layer)
        assert torch.all(linear_layer.weight != 0).item()

    def test_initialize_weights_sequential(self):
        # Create a sequential model with 3 linear layers
        sequential_model = nn.Sequential(
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100)
        )

        # Initialize the weights and bias to zeros
        for layer in sequential_model:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        _init_weights(sequential_model)
        for layer in sequential_model:
            assert torch.all(layer.weight != 0).item()

    def test_config_instance_variables(self):
        model_configuration = {
            "backbone": "resnet18",
            "pretrained": True,
            "freeze": True,
            "cut": -1,
            "backbone_output_dim": 512,
            "out_features": 100,
            "dropout": 0.5,
            "hidden_units": 512,
            "device": "cuda:0"
        }
        reidentify_model = ReidentifyModel(model_configuration)
        assert reidentify_model.backbone_name == "resnet18"
        assert reidentify_model.pretrained == True
        assert reidentify_model.freeze == True
        assert reidentify_model.cut == -1
        assert reidentify_model.backbone_output_dim == 512
        assert reidentify_model.out_features == 100
        assert reidentify_model.dropout == 0.5
        assert reidentify_model.hidden_units == 512
        assert reidentify_model.device == torch.device("cuda:0")


if __name__ == '__main__':
    unittest.main()
