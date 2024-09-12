import torch
from pathlib import Path

from ikepono.configuration import Configuration
from ikepono.hardtripletsampler import HardTripletBatchSampler
from ikepono.labeledimagedataset import LabeledImageDataset
from ikepono.reidentifymodel import ReidentifyModel
from ikepono.vectorstore import VectorStore

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import mlfow

class Reidentifier:
    @classmethod
    def for_training(cls, configuration : Configuration):
        reidentifier = Reidentifier(configuration)
        lies = reidentifier.model.build_labeled_image_embeddings(reidentifier.train_dataset, reidentifier.dataset_device)
        reidentifier.vector_store.initialize(lies)
        reidentifier.sampler.initialize(reidentifier.vector_store)
        return reidentifier

    @classmethod
    def for_inference(cls, configuration : Configuration):
        raise "Not implemented"


    def __init__(self, configuration : Configuration):
        assert configuration is not None, "configuration cannot be None"

        self.configuration = configuration
        configuration.save("reidentifier_train_configuration.json")
        data_root_dir = configuration.train_configuration()["data_path"]
        training_dir = Path(data_root_dir) / "train"
        validation_dir = Path(data_root_dir) / "valid"
        self.dataset_device = self.configuration.train_configuration()["dataset_device"]
        self.train_dataset = LabeledImageDataset.from_directory(training_dir,
                                                                device=self.dataset_device)
        self.sampler = HardTripletBatchSampler(self.train_dataset, configuration.train_configuration()["n_triplets"])

        self.validation_dataset = LabeledImageDataset.from_directory(validation_dir,
                                                                     device=self.dataset_device)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                  batch_size=self.configuration.train_configuration()["n_triplets"],
                                                  shuffle=True,
                                                  collate_fn=LabeledImageDataset.collate)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset,
                                                       batch_size=self.configuration.train_configuration()["n_triplets"],
                                                       shuffle=False,
                                                       collate_fn=LabeledImageDataset.collate)
        self.num_epochs = self.configuration.train_configuration()["epochs"]

        self.model_device = self.configuration.train_configuration()["model_device"]
        self.model = ReidentifyModel(configuration.model_configuration(),
                                     configuration.train_configuration(),
                                     self.train_dataset.num_classes(),
                                     self.model_device)


        self.embedding_dimension = configuration.model_configuration()["output_vector_size"]
        self.vector_store = VectorStore(self.embedding_dimension)

        self.artifacts_path = configuration.model_configuration()["artifacts_path"]

    def train(self):
        experiment_id = self.configure_mlflow()
        num_epochs = self.num_epochs

        with mlflow.start_run() as active_run:
            mlflow.log_params(configuration.model_configuration())
            mlflow.log_params(configuration.train_configuration())
            mlflow.log_param("total_classes", len(set(self.train_dataset.labels)))
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("run_id", active_run.info.run_id)

            mlflow.log_artifact("reidentifier_train_configuration.json")
            start = time.time()

            assert False, "I need to move this ptml_train code into model.train"
            best_mrr = 0.0
            for epoch in range(1, num_epochs + 1):
                batch_lossses = self._train_all_batches(self.model, loss_func, model.device, train_loader, optimizer, loss_optimizer,
                                            epoch)
                train_losses.extend(batch_lossses)
                accuracies = _ptml_test(train_set, val_set, model, accuracy_calculator)
                mrrs.append(accuracies["mean_reciprocal_rank"])
                step = epoch * len(train_loader)
                # print(f"Step {step} MRR: {accuracies['mean_reciprocal_rank']}")
                mlflow.log_metric("mean_reciprocal_rank", accuracies["mean_reciprocal_rank"], step=step)

    def _train_all_batches(self, model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch):
        batch_losses = []
        model.train()
        batch_count = len(train_loader)
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_optimizer.zero_grad()
            embeddings = model(data)
            loss = loss_func(embeddings, labels)
            loss.backward()
            optimizer.step()
            loss_optimizer.step()
            batch_losses.append(loss.item())
            mlflow.log_metric("batch_loss", loss.item(), step=epoch * batch_count + batch_idx)
            if batch_idx % 100 == 0:
                print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
        return batch_losses

    def _get_all_embeddings(self, dataset, model):
        stripped_dataset = [(lit.image, lit.label) for lit in dataset]

        tester = testers.BaseTester()
        return tester.get_all_embeddings(stripped_dataset, model)

    ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
    def _test_accuracy(train_set, test_set, model, accuracy_calculator):
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

    def configure_mlflow(self):
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


