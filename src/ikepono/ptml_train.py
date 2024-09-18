import time
import torch
from torch import optim
from torchvision import transforms as xform

import mlflow
import numpy as np
import os
from ikepono.configuration import Configuration
from ikepono.reidentifymodel import ReidentifyModel
from ikepono.splittableimagedataset import SplittableImageDataset
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from typing import Union


def _ptml_train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch):
    batch_losses = []
    model._reidentify_model_train()
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
        mlflow.log_metric("batch_loss", loss.item(), step=epoch*batch_count + batch_idx)
        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
    return batch_losses

### convenient function from pytorch-metric-learning ###
def _get_all_embeddings(dataset, model):
    stripped_dataset = [(lit.image, lit.label_idx) for lit in dataset]

    tester = testers.BaseTester()
    return tester.get_all_embeddings(stripped_dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def _ptml_test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = _get_all_embeddings(train_set, model)
    test_embeddings, test_labels = _get_all_embeddings(test_set, model)
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


def _ptml_collate(labeledImageTensors: list["LabeledImageTensor"]) -> dict[str, Union[torch.Tensor, np.ndarray]]:
    images = torch.stack([item.image for item in labeledImageTensors])
    labels = torch.Tensor([int(item.label_idx) for item in labeledImageTensors]).type(torch.int64)
    # labels = torch.stack(tensor)
    sources = np.array([item.source for item in labeledImageTensors])

    return images, labels

def build_and_train(configuration):
    img_mean, img_std = (0.1307,), (0.3081,)
    transforms = xform.Compose([
        xform.ToTensor(),
        xform.Resize((224, 224)),
        xform.Normalize(img_mean, img_std)
    ])
    batch_size = 32

    data_dir = configuration["train"]["train_data_path"]
    mlflow_data_dir = "../../data"
    db_path = os.path.join(mlflow_data_dir, 'mlflow.db')
    artifacts_path = configuration["model"]["artifacts_path"]

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)

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

    dataset = SplittableImageDataset.from_directory(root_dir=data_dir, k=10)
    total_classes = len(set(dataset.labels))
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=_ptml_collate
    )
    model = ReidentifyModel(configuration["model"], configuration["train"], total_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = configuration["train"]["epochs"]

    loss_func = losses.SubCenterArcFaceLoss(num_classes=51, embedding_size=128).to(model.device)
    loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1","mean_reciprocal_rank","mean_average_precision_at_r"), k="max_bin_count")

    with mlflow.start_run() as active_run:
        mlflow.log_params(configuration["model"])
        mlflow.log_params(configuration["train"])
        mlflow.log_param("total_classes", total_classes)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_mean", img_mean)
        mlflow.log_param("img_std", img_std)
        mlflow.log_param("run_id", active_run.info.run_id)

        mlflow.log_artifact("ptml_configuration.json")
        start = time.time()
        mrrs = []
        train_losses = []
        best_mrr = 0.0
        for epoch in range(1, num_epochs+1):
            batch_lossses = _ptml_train(model, loss_func, model.device, train_loader, optimizer, loss_optimizer, epoch)
            train_losses.extend(batch_lossses)
            accuracies = _ptml_test(train_set, val_set, model, accuracy_calculator)
            mrrs.append(accuracies["mean_reciprocal_rank"])
            step = epoch * len(train_loader)
            #print(f"Step {step} MRR: {accuracies['mean_reciprocal_rank']}")
            mlflow.log_metric("mean_reciprocal_rank", accuracies["mean_reciprocal_rank"], step=step)
            mlflow.log_metric("precision_at_1", accuracies["precision_at_1"], step=step)
            mlflow.log_metric("mean_average_precision_at_r", accuracies["mean_average_precision_at_r"], step=step)
            if accuracies["mean_reciprocal_rank"] > best_mrr:
                best_mrr = accuracies["mean_reciprocal_rank"]
                model_name = f"{experiment_name}_{experiment_id}.pth"
                torch.save(model.state_dict(), model_name)
                mlflow.log_artifact(f"{model_name}", model_name)
                print(f"Best model saved with MRR {best_mrr} to {model_name}")
        end = time.time()
        print(f"Training {num_epochs} epochs took {end - start} seconds")
        print(f"Mean Reciprocal Rank: {mrrs}")
        print(f"Losses: {train_losses}")


if __name__ == "__main__":
    configuration = Configuration("ptml_configuration.json")
    build_and_train(configuration)