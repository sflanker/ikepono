import time
import torch
import torch.nn as nn

import ikepono.vectorstore as VectorStore
import mlflow
import numpy as np
import timm
from ikepono.labeledimagedataset import LabeledImageDataset
from ikepono.labeledimageembedding import LabeledImageEmbedding
from pathlib import Path
from pytorch_metric_learning import losses


def _init_weights(model: nn.Module) -> None:
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None and model.bias.data is not None:
            model.bias.data.fill_(0.01)
    elif isinstance(model, nn.Sequential):
        for layer in model:
            _init_weights(layer)

class ReidentifyModel(nn.Module):
    def __init__(self, model_configuration, train_configuration, num_known_individuals : int):
        super(ReidentifyModel, self).__init__()
        self.backbone_name = model_configuration["backbone"]
        self.pretrained = model_configuration["pretrained"]
        self.freeze = model_configuration["freeze"]
        self.cut = model_configuration["cut"]
        self.backbone_output_dim = model_configuration["backbone_output_dim"]
        self.output_vector_size = model_configuration["output_vector_size"]
        self.dropout = model_configuration["dropout"]
        self.hidden_units = model_configuration["hidden_units"]
        self.device = model_configuration["model_device"]

        self.backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained)
        if not self.pretrained:
            _init_weights(self.backbone)
        else:
            self.backbone.requires_grad_(not self.freeze)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:self.cut]).to(self.device)
        self.projection = self._create_head(self.backbone_output_dim, self.hidden_units, self.output_vector_size).to(self.device)
        _init_weights(self.projection)

        self._initialize_training_parameters(
            lr=train_configuration["learning_rate"],
            optimizer_str=train_configuration["optimizer"],
            criterion_str=train_configuration["criterion"],
            embedding_size=self.output_vector_size,
            num_classes=num_known_individuals
        )

    def _initialize_training_parameters(self, lr : float, optimizer_str: str, criterion_str : str, embedding_size: int, num_classes: int) -> None:
        if optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_str}")

        if criterion_str == "SubCenterArcFaceLoss":
            self.criterion = losses.SubCenterArcFaceLoss(num_classes = num_classes, embedding_size = embedding_size)

    def forward(self, img_tensors: torch.Tensor) -> torch.Tensor:
        backbone_representation = self.backbone(img_tensors)
        embedding = self.projection(backbone_representation)

        return embedding

    def _create_head(self, backbone_output_dim, hidden_units, output_vector_size):
         return nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_units),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units, output_vector_size)
        )

    # Note: This cannot be called `train`, because that's a member variable in base `Module` class
    def _reidentify_model_train(self, vector_store : VectorStore, dataloader, num_epochs : int):
        #TODO assert all needed parameters are set
        #TODO deprecated? The main training loop is now in Reidentifier
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.device is not None
        assert vector_store is not None
        assert dataloader is not None
        assert dataloader.batch_sampler.get_initialized() == True
        assert vector_store.get_initialized() == True

        # Make sure label counts all match up
        assert len(vector_store.labels) == len(dataloader.dataset.idx_to_label)

        # Training loop
        best_loss = float('inf')

        losses = []
        epoch = 1
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs}")
            loss = self._train_one_epoch(dataloader, vector_store)
            losses.append(loss)
            # Save best model
            if loss < best_loss:
                best_loss = loss
                self.save_model()
                print(f"Best model saved with loss {best_loss}")
        return losses

    def save_model(self, filename : str = "best_model.pth"):
        torch.save(self.state_dict(), filename)
        maybe_run = mlflow.active_run()
        if maybe_run is not None:
            mlflow.log_artifact(filename)

    def load_weights(self, filename : str):
        self.load_state_dict(torch.load(filename))


    def build_labeled_image_embeddings(self, dataset : LabeledImageDataset, device) -> np.ndarray[LabeledImageEmbedding]:
        livs = []
        for lit in dataset:
            # Unsqueeze for model, which expects a batch dimension
            tensor = self.forward(lit.image.to(self.device).unsqueeze(0))
            # Grab the embedding from the tensor, which is a 1xN tensor
            embedding = tensor.detach().cpu().numpy()[0] #TODO: Hardcoded 'cpu()' ok? I think so.
            label = dataset.idx_to_label[lit.label_idx]
            source = Path(lit.source)
            livs.append(LabeledImageEmbedding(embedding, label, source, lit.label_idx))
        return np.array(livs)

    def _train_one_epoch(self, train_loader, vector_store):
        start = time.time()
        i = 0
        epoch_loss = 0.0
        assert len(train_loader) > 0
        for batch in train_loader:
            if i % 10 == 0:
                print(".", end="")
            assert len(batch['images']) == 6, f"Expected batch size 6, got {len(batch['images'])}"
            i += 1
            loss = self._train_one_batch(batch=batch, vector_store=vector_store)
            epoch_loss += loss.item()
        end = time.time()
        self._log(f"Epoch took {end - start} seconds")
        return epoch_loss

    def _log(self, message):
        print(message)

    def _train_one_batch(self, batch : dict[str, any], vector_store):
        # Get the actual data for these indices (move from dataloader device to model device)
        image_tensors_dl = batch["images"]
        label_idxs = batch["label_indexes"]
        sources = batch["sources"]

        batch_size = len(label_idxs)
        image_tensors = torch.stack([image_tensors_dl[i] for i in range(batch_size)]).to(self.device)
        label_tensor = label_idxs.to(self.device)
        # Assert they are the same batch size
        assert len(image_tensors) == len(label_tensor)

        # Forward pass
        embeddings = self.forward(image_tensors)
        # Compute loss
        assert embeddings.shape == (batch_size, self.output_vector_size)
        assert label_tensor.shape == (batch_size,)

        loss = self.criterion(embeddings, label_tensor)
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for i, source in enumerate(sources):
            label = vector_store.label_for_source(Path(source))
            assert label == Path(source).parent.name
            vector_store.update_or_add_vector(Path(source), embeddings[i].detach().cpu().numpy(), label)

        return loss