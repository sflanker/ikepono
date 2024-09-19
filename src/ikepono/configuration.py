import torch

import json
from pathlib import Path
from typing import Optional


#TODO: Both train and model configurations have device. That should be a single value. More of a model thing. But maybe I should rename model configuration to reidentifier configuration.
#TODO: Move mlflow configuration values into here (tracking_uri, experiment_name, artifacts_path)
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
        assert "inference" in configuration, "inference is missing"
        # TODO: Add validation back in before release
        #
        # assert "epochs" in configuration["train"], "epochs is missing from train configuration"
        # assert "learning_rate" in configuration["train"], "learning_rate is missing from train configuration"
        # assert "optimizer" in configuration["train"], "optimizer is missing from train configuration"
        # assert "criterion" in configuration["train"], "criterion is missing from train configuration"
        # assert "dataset_device" in configuration["train"], "dataset device is missing from train configuration"
        # assert "model_device" in configuration["train"], "model device is missing from train configuration"
        # assert "data_path" in configuration["train"], "data_path is missing from train configuration"
        # assert "k" in configuration["train"], "k is missing from train configuration"
        # assert "n_triplets" in configuration["train"], "n_triplets is missing from train configuration"
        #
        # assert "backbone" in configuration["model"], "backbone is missing from model configuration"
        # assert "pretrained" in configuration["model"], "pretrained is missing from model configuration"
        # assert "freeze" in configuration["model"], "freeze is missing from model configuration"
        # assert "cut" in configuration["model"], "cut is missing from model configuration"
        # assert "dropout" in configuration["model"], "dropout is missing from model configuration"
        # assert "hidden_units" in configuration["model"], "hidden_units is missing from model configuration"
        # assert "output_vector_size" in configuration["model"], "output_vector_size is missing from model configuration"
        # assert "dataset_device" in configuration["model"], "dataset device is missing from model configuration"
        # assert "model_device" in configuration["model"], "model device is missing from model configuration"
        # assert "artifacts_path" in configuration["model"], "artifacts path is missing from model configuration"
        #
        # assert 2 == len(configuration), "Configuration should have exactly two keys: model and train"
        # assert 9 == len(configuration["train"]), f"Train configuration should have exactly 9 keys, not {len(configuration['train'])}: epochs, learning_rate, optimizer, criterion, dataset_device, model_device, train_data_path, k, n_triplets"
        # assert 11 == len(configuration["model"]), "Model configuration should have exactly 10 keys: backbone, pretrained, freeze, cut, dropout, backbone_output_dim, hidden_units, output_vector_size, dataset_device, model_device, artifacts_path"

    @staticmethod
    def defaults():
        configuration = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
        configuration["model"] = {"backbone": "resnet18", "pretrained": True, "freeze": True, "cut": -1, "dropout": 0.5,
                              "backbone_output_dim": 512, "hidden_units": 512, "output_vector_size": 128, "dataset_device": torch.device("cpu"),
                                  "model_device" : device, "artifacts_path": "./artifacts"}

        configuration["train"] = {"epochs": 10, "learning_rate": 0.001,
                              "optimizer" : "adam", "criterion" : "SubCenterArcFaceLoss", "dataset_device": torch.device("cpu"),
                                  "model_device" : device, "data_path": "/mnt/d/scratch_data/mantas/train_valid/kona",
                                  "k" : 5, "n_triplets" : 32}

        configuration["inference"] = {
            "weights_path": "./artifacts/weights.pth",
            "vector_store_path": "./artifacts/vector_store.pth",
            "id_to_manta_id_path": "./artifacts/id_to_manta_id.json",
            "id_to_source_path": "./artifacts/id_to_source.json",
            "manta_id_to_name_path": "./artifacts/manta_id_to_name.json",
        }
        return configuration

    def load(self, config_file : Path):
        with open(config_file, 'r') as f:
            configuration = json.load(f)
        return configuration

    def save(self, config_file : Path):
        # Check if the device is a torch device and, if so, save it's name
        dataset_device_name = self.configuration["train"]["dataset_device"].type if isinstance(self.configuration["train"]["dataset_device"], torch.device) else self.configuration["train"]["dataset_device"]
        model_device_name = self.configuration["train"]["model_device"].type if isinstance(self.configuration["train"]["model_device"], torch.device) else self.configuration["train"]["model_device"]
        self.configuration["train"]["dataset_device"] = dataset_device_name
        self.configuration["train"]["model_device"] = model_device_name
        with open(config_file, 'w') as f:
            json.dump(self.configuration, f)
