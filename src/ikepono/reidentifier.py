import time
import torch
from torch import optim
from torch.utils.data import DataLoader

import mlflow
import numpy as np
import os
from ikepono.configuration import Configuration
from ikepono.hardtripletsampler import HardTripletBatchSampler
from ikepono.indexedimagetensor import IndexedImageTensor
from ikepono.labeledimagedataset import LabeledImageDataset
from ikepono.reidentifymodel import ReidentifyModel
from ikepono.vectorstore import VectorStore
from pathlib import Path
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


class Reidentifier:
    _built_from_factory = False

    @classmethod
    def _factory_built(cls, *args, **kwargs):
        instance = cls.__new__(cls)
        instance._built_from_factory = True
        instance.__init__(*args, **kwargs)
        return instance

    @classmethod
    def _start_mlflow_run(cls, configuration: Configuration):
        mlflow_data_dir = "../../data"
        db_path = os.path.join(mlflow_data_dir, 'mlflow.db')
        artifacts_path = configuration.model_configuration()["artifacts_path"]

        # Set MLflow tracking URI to use SQLite
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        os.environ['MLFLOW_ARTIFACTS_PATH'] = artifacts_path

        # Optional: Set up a new experiment
        experiment_name = "ikepono"
        try:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifacts_path)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    @classmethod
    def for_training(cls, configuration : Configuration):
        reidentifier = Reidentifier._factory_built(configuration)
        configuration.save("reidentifier_train_configuration.json")
        maybe_run = mlflow.active_run()
        if maybe_run is None:
            Reidentifier._start_mlflow_run(configuration)
        mlflow.log_artifact("reidentifier_train_configuration.json")
        lies = reidentifier.model.build_labeled_image_embeddings(reidentifier.train_dataset, reidentifier.dataset_device)
        reidentifier.vector_store.initialize(lies)
        reidentifier.sampler.initialize(reidentifier.vector_store)
        return reidentifier

    @classmethod
    def for_inference(cls, configuration : Configuration):
        reidentifier = Reidentifier(configuration)
        raise "Not implemented"
        # TODO: Load the model from the artifacts path
        # TODO: Load the vector store from the artifacts path
        # TODO: Load the vector store ID -> manta_id dictionary from the artifacts path


    def __init__(self, configuration : Configuration):
        assert configuration is not None, "configuration cannot be None"
        if not hasattr(self, '_built_from_factory'):
            raise Exception("This class cannot be instantiated directly. Use the factory methods `for_training` or `for_inference`.")
        self._built_from_factory = True

        self.configuration = configuration
        data_root_dir = configuration.train_configuration()["data_path"]
        training_dir = Path(data_root_dir) / "train"
        validation_dir = Path(data_root_dir) / "valid"
        self.dataset_device = self.configuration.train_configuration()["dataset_device"]
        self.train_dataset = LabeledImageDataset.from_directory(training_dir,
                                                                device=self.dataset_device, k=configuration.train_configuration()["k"])
        self.sampler = HardTripletBatchSampler(self.train_dataset, configuration.train_configuration()["n_triplets"])

        self.validation_dataset = LabeledImageDataset.from_directory(validation_dir,
                                                                     device=self.dataset_device, k=configuration.train_configuration()["k"])

        self.train_loader = DataLoader(self.train_dataset,
                                                        batch_size=self.configuration.train_configuration()["n_triplets"],
                                                        shuffle=True,
                                                        collate_fn=IndexedImageTensor.collate)
        self.validation_loader = DataLoader(self.validation_dataset,
                                                             batch_size=self.configuration.train_configuration()["n_triplets"],
                                                             shuffle=False,
                                                             collate_fn=IndexedImageTensor.collate)
        self.num_epochs = self.configuration.train_configuration()["epochs"]

        self.model_device = self.configuration.train_configuration()["model_device"]
        num_known_individuals = len(set(self.train_dataset.labels))
        self.model = ReidentifyModel(configuration.model_configuration(),
                                     configuration.train_configuration(),
                                     num_known_individuals)


        self.embedding_dimension = configuration.model_configuration()["output_vector_size"]
        self.vector_store = VectorStore(self.embedding_dimension)

        self.artifacts_path = configuration.model_configuration()["artifacts_path"]

    def train(self) -> tuple[float, float]:
        experiment_id = self._configure_mlflow(self.configuration)
        num_epochs = self.num_epochs

        # Check if mlflow has an active run. If so, stop it
        maybe_run = mlflow.active_run()
        if maybe_run is not None:
            mlflow.end_run()

        with mlflow.start_run() as active_run:
            mlflow.log_params(self.configuration.model_configuration())
            mlflow.log_params(self.configuration.train_configuration())
            mlflow.log_param("total_classes", len(set(self.train_dataset.labels)))
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("run_id", active_run.info.run_id)

            mlflow.log_artifact("reidentifier_train_configuration.json")
            # TODO: params from configuration (num_classes from dataset)
            loss_func = losses.SubCenterArcFaceLoss(num_classes=61, embedding_size=128).to(self.model.device)
            loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            mlflow.log_param("loss_func", "SubCenterArcFaceLoss")
            mlflow.log_param("loss_optimizer", "Adam")
            mlflow.log_param("loss_optimizer_lr", 1e-4)
            mlflow.log_param("optimizer_lr", 0.001)
            mlflow.log_param("n_triplets", self.configuration.train_configuration()["n_triplets"])
            mlflow.log_param("n_batches_trainset", len(self.train_loader) / (self.configuration.train_configuration()["n_triplets"]))
            mlflow.log_param("n_batches_validationset", len(self.validation_loader) / (self.configuration.train_configuration()["n_triplets"]))
            accuracy_calculator = AccuracyCalculator(
                include=("precision_at_1", "mean_reciprocal_rank", "mean_average_precision_at_r"), k="max_bin_count")

            start = time.time()
            best_mrr = 0.0
            for epoch in range(1, num_epochs + 1):
                train_start = time.time()
                batch_losses = self._train_all_batches(self.model, loss_func, self.model.device, self.train_loader, optimizer, loss_optimizer,
                                            epoch)
                # Calculate validation loss
                val_loss = self._calculate_validation_loss(self.model, loss_func, self.model.device,
                                                           self.validation_loader)
                batch_count = len(self.validation_loader)
                mlflow.log_metric("val_loss", val_loss, step=epoch * batch_count)
                # Ratio of validation loss to training loss
                loss_ratio = val_loss / np.mean(batch_losses)
                mlflow.log_metric("val_loss_ratio", loss_ratio, step=epoch * batch_count)
                print(f"Epoch {epoch} Validation Loss: {val_loss} Loss Ratio: {loss_ratio}")


                accuracies = self._test_accuracy(self.train_dataset, self.validation_dataset, self.model, accuracy_calculator)
                step = epoch * len(self.train_loader)
                # print(f"Step {step} MRR: {accuracies['mean_reciprocal_rank']}")
                mlflow.log_metric("mean_reciprocal_rank", accuracies["mean_reciprocal_rank"], step=step)
                epoch_mrr = accuracies["mean_reciprocal_rank"]
                if epoch_mrr > best_mrr:
                    best_mrr = epoch_mrr
                    self.model.save_model(self.artifacts_path + f"/model_{experiment_id}.pth")
                train_end = time.time()
                print(f"Epoch {epoch} took {train_end - train_start:.1f} seconds")
            return time.time() - start, best_mrr

    def _train_all_batches(self, model : ReidentifyModel, loss_func : callable, device : torch.device, train_loader : DataLoader, optimizer, loss_optimizer, epoch:int) -> list[float]:
        #TODO: Isn't model.device superior to device as a param?
        batch_losses = []
        #model._train()
        batch_count = len(train_loader)
        for batch_idx, loaded_data in enumerate(train_loader):
            img_tensor = loaded_data["images"]
            label_idxs = loaded_data["label_indexes"]
            img_tensor, label_idxs = img_tensor.to(device), label_idxs.to(device)
            optimizer.zero_grad()
            loss_optimizer.zero_grad()
            embeddings = model(img_tensor)
            loss = loss_func(embeddings, label_idxs)
            loss.backward()
            optimizer.step()
            loss_optimizer.step()
            batch_losses.append(loss.item())
            mlflow.log_metric("batch_loss", loss.item(), step=epoch * batch_count + batch_idx)
            if batch_idx % 10 == 0:
                print("Epoch {} Batch {}: Loss = {}".format(epoch, batch_idx, loss))
        return batch_losses

    def _get_all_embeddings(self, dataset, model):
        stripped_dataset = [(lit.image, lit.label_idx) for lit in dataset]

        tester = testers.BaseTester()
        return tester.get_all_embeddings(stripped_dataset, model)

    def _calculate_validation_loss(self, model, loss_func, device, validation_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for loaded_data in validation_loader:
                img_tensor = loaded_data["images"]
                label_idxs = loaded_data["label_indexes"]
                img_tensor, label_idxs = img_tensor.to(device), label_idxs.to(device)
                embeddings = model(img_tensor)
                loss = loss_func(embeddings, label_idxs)
                total_loss += loss.item()
        model.train()
        return total_loss / len(validation_loader)

    ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
    def _test_accuracy(self, train_set, test_set, model, accuracy_calculator):
        train_embeddings, train_labels = self._get_all_embeddings(train_set, model)
        test_embeddings, test_labels = self._get_all_embeddings(test_set, model)
        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        print("Computing accuracy")
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, test_labels, train_embeddings, train_labels, False
        )
        print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
        print("Mean reciprocal rank = {}".format(accuracies["mean_reciprocal_rank"]))
        print("Mean average precision at r = {}".format(accuracies["mean_average_precision_at_r"]))
        return accuracies

    def _configure_mlflow(self, configuration: Configuration):
        #TODO: This should be controllged by the configuration
        mlflow_data_dir = "../../data"
        db_path = os.path.join(mlflow_data_dir, 'mlflow.db')
        artifacts_path = self.artifacts_path
        # Set MLflow tracking URI to use SQLite
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        os.environ['MLFLOW_ARTIFACTS_PATH'] = artifacts_path

        # Optional: Set up a new experiment
        experiment_name = "ikepono"
        try:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifacts_path)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)
        return experiment_id


