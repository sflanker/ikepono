import unittest
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import os

from src.ikepono.configuration import Configuration

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.ReidentifyModel import _init_weights, ReidentifyModel
from test_hardtripletsampler import SamplerTests
from test_splittableimagedataset import SplittableImageDatasetTests
class ReidentifyModelTests(unittest.TestCase):
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
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
            "output_vector_size": 100,
            "dropout": 0.5,
            "hidden_units": 512,
            "device": self.torch_device
        }
        reidentify_model = ReidentifyModel(model_configuration)
        assert reidentify_model.backbone_name == "resnet18"
        assert reidentify_model.pretrained == True
        assert reidentify_model.freeze == True
        assert reidentify_model.cut == -1
        assert reidentify_model.backbone_output_dim == 512
        assert reidentify_model.output_vector_size == 100
        assert reidentify_model.dropout == 0.5
        assert reidentify_model.hidden_units == 512
        assert reidentify_model.output_vector_size == 100
        assert reidentify_model.device == torch.device(model_configuration["device"])

    def test_forward(self):
        model_configuration = {
            "backbone": "resnet18",
            "pretrained": True,
            "freeze": True,
            "cut": -1,
            "backbone_output_dim": 512,
            "output_vector_size": 100,
            "dropout": 0.5,
            "hidden_units": 512,
            "device": self.torch_device
        }
        reidentify_model = ReidentifyModel(model_configuration)
        img_tensor = torch.randn(32, 3, 224, 224).to(torch.device(model_configuration["device"]))
        embedding = reidentify_model(img_tensor)
        assert embedding.shape == (32, 100)

    def test_train(self):
        return False

    def test_train_one_epoch(self):
        return False

    def test_train_one_batch(self):
        model, dataloader, vector_store = self.simple_model()
        model._train_one_batch(dataloader, batch_indices, vector_store)
        return False

    def simple_model(self):
        configuration = Configuration("test_configuration.json")
        sampler = SamplerTests.simple_sampler()
        ds = SplittableImageDatasetTests.simple_dataset()
        loader = DataLoader(ds, batch_size=32, sampler=sampler)
        model = ReidentifyModel(configuration)
        return model, loader, vector_store


if __name__ == '__main__':
    unittest.main()