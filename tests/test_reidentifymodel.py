import unittest
from dataclasses import fields

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from pytorch_metric_learning import losses

from ikepono.configuration import Configuration
from ikepono.hardtripletsampler import HardTripletBatchSampler
from ikepono.labeledimageembedding import LabeledImageEmbedding, LabeledImageTensor
from ikepono.vectorstore import VectorStore
from test_hardtripletsampler import SamplerTests
from test_splittableimagedataset import SplittableImageDatasetTests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.reidentifymodel import _init_weights, ReidentifyModel
from src.ikepono.splittableimagedataset import SplittableImageDataset

class ReidentifyModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
        cls.dataset_device = torch.device("cpu")
        cls.data_dir = Path("/mnt/d/scratch_data/mantas/by_name/original/kona")


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
            "model_device": self.model_device,
            "dataset_device": self.dataset_device
        }
        train_configuration = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "optimizer": "adam",
            "criterion": "SubCenterArcFaceLoss",
            "num_classes": 10,
            "weight_decay": 0.0001,
            "dataset_device": self.dataset_device,
            "model_device": self.model_device
        }

        reidentify_model = ReidentifyModel(model_configuration, train_configuration)
        assert reidentify_model.backbone_name == "resnet18"
        assert reidentify_model.pretrained == True
        assert reidentify_model.freeze == True
        assert reidentify_model.cut == -1
        assert reidentify_model.backbone_output_dim == 512
        assert reidentify_model.output_vector_size == 100
        assert reidentify_model.dropout == 0.5
        assert reidentify_model.hidden_units == 512
        assert reidentify_model.output_vector_size == 100
        assert reidentify_model.device == torch.device(model_configuration["model_device"])

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
            "model_device": self.model_device
        }

        train_configuration = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "optimizer": "adam",
            "criterion": "SubCenterArcFaceLoss",
            "num_classes": 10,
            "weight_decay": 0.0001,
            "dataset_device": self.dataset_device,
            "model_device": self.model_device
        }
        reidentify_model = ReidentifyModel(model_configuration, train_configuration)
        img_tensor = torch.randn(32, 3, 224, 224).to(torch.device(model_configuration["model_device"]))
        embedding = reidentify_model(img_tensor)
        assert embedding.shape == (32, 100)

    def test_train(self):
        model, loader, vs = self.simple_model()
        model.device = torch.device("cpu")
        model.to("cpu")
        model.backbone = model.backbone.to("cpu")
        model.projection = model.projection.to("cpu")
        assert vs.get_initialized() == True
        loader.batch_sampler.initialize(vs)

        loss = model.train(dataloader=loader, vector_store=vs, num_epochs=20)
        assert loss >= 0, f"Expected non-negative loss, got {loss}" # With simple data, can get loss = 0.0 by memorizing input

    def test_train_one_epoch(self):
        model, loader, vs = self.simple_model()
        model.device = torch.device("cpu")
        model.to("cpu")
        model.backbone = model.backbone.to("cpu")
        model.projection = model.projection.to("cpu")

        il = iter(loader)
        loss = model._train_one_epoch(il, vs)
        assert loss >= 0, f"Expected nonnegative loss, got {loss}" # Loss can be 0.0 with simple data

    def test_train_one_batch(self):
        model, loader, vs,  = self.simple_model()
        assert vs.get_initialized(), "Vector store not initialized"
        assert loader.batch_sampler.get_initialized(), "Sampler not initialized"

        model.device = torch.device("cpu")
        model.to("cpu")
        model.backbone = model.backbone.to("cpu")
        model.projection = model.projection.to("cpu")

        il = iter(loader)
        batch = next(il) # Uses sampler
        batch["images"] = batch["images"].to("cpu")
        loss = model._train_one_batch(batch, vs)
        assert loss.item() > 0, f"Expected loss > 0, got {loss.item()}"

    def test_real_dataset(self):
        print("Running test_real_dataset. Remove this from the test suite after train debugged.")
        configuration = Configuration("test_configuration.json")
        individuals_per_batch = 4
        vector_store = VectorStore(dimension=configuration.model_configuration()["output_vector_size"])

        dataset = SplittableImageDataset.from_directory(root_dir=self.data_dir, k=10)
        # This isn't a configuration value!
        configuration.configuration["train"]["num_classes"] = len(set(dataset.labels))
        print("Built dataset")
        sampler = HardTripletBatchSampler(dataset, 3)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            collate_fn=LabeledImageTensor.collate)
        print("Built loader")

        model = ReidentifyModel(configuration.model_configuration(), configuration.train_configuration())
        print("Built model")
        vector_store.initialize(model.build_labeled_image_embeddings(dataset, torch.device("cpu")))
        sampler.initialize(vector_store)

        print("Beginning train")
        # Note that this will take a long time to run
        losses = model.train(dataloader=loader, vector_store=vector_store, num_epochs=250)
        print("Finished train")
        for loss in losses:
            print(f"{loss:.1f}", end=", ")
        assert False

    def simple_model(self):
        configuration = Configuration("test_configuration.json")
        ds = SplittableImageDatasetTests.simple_dataset()
        embedding_size = configuration.model_configuration()["output_vector_size"]
        vs = VectorStore(dimension=embedding_size)

        num_classes = len(set(ds.labels))

        sampler = HardTripletBatchSampler(ds, 3)

        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=LabeledImageTensor.collate)
        model = ReidentifyModel(configuration.model_configuration(), configuration.train_configuration())
        vs.initialize(model.build_labeled_image_embeddings(ds, torch.device("cpu")))
        sampler.initialize(vs)

        return model, loader, vs

    def test_all_properties_assigned(self):
        attributes_expected = ["backbone_name", "pretrained", "freeze", "cut", "backbone_output_dim",
                               "output_vector_size", "dropout", "hidden_units", "device"]
        configuration = Configuration("test_configuration.json")
        model = ReidentifyModel(configuration.model_configuration(), configuration.train_configuration())
        for attribute in attributes_expected:
            assert hasattr(model, attribute), f"Expected attribute {attribute} not found in ReidentifyModel"
            assert getattr(model, attribute) is not None, f"Expected attribute {attribute} to be assigned in ReidentifyModel"


if __name__ == '__main__':
    unittest.main()