import torch.nn as nn
import timm
import torch
import ikepono.VectorStore as VectorStore

def _init_weights(model: nn.Module) -> None:
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None and model.bias.data is not None:
            model.bias.data.fill_(0.01)
    elif isinstance(model, nn.Sequential):
        for layer in model:
            _init_weights(layer)

class ReidentifyModel(nn.Module):
    def __init__(self, model_configuration):
        super(ReidentifyModel, self).__init__()
        self.backbone_name = model_configuration["backbone"]
        self.pretrained = model_configuration["pretrained"]
        self.freeze = model_configuration["freeze"]
        self.cut = model_configuration["cut"]
        self.backbone_output_dim = model_configuration["backbone_output_dim"]
        self.output_vector_size = model_configuration["output_vector_size"]
        self.dropout = model_configuration["dropout"]
        self.hidden_units = model_configuration["hidden_units"]
        self.device = torch.device(model_configuration["device"])

        self.backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained)
        if not self.pretrained:
            _init_weights(self.backbone)
        else:
            self.backbone.requires_grad_(not self.freeze)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:self.cut]).to(self.device)
        self.projection = self._create_head(self.backbone_output_dim, self.hidden_units, self.output_vector_size).to(self.device)
        _init_weights(self.projection)

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

    def train(self, vector_store : VectorStore, dataloader, num_epochs : int, criterion, optimizer, scheduler):
        # Training loop
        best_loss = float('inf')

        initial_embeddings, initial_labels = self.get_embeddings(vector_store, dataloader, self.device)
        vector_store.add_with_ids(initial_embeddings, initial_labels)

        for epoch in range(num_epochs):
            loss = self._train_one_epoch(vector_store)
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def _train_one_epoch(self, train_loader, vector_store):
        for batch_indices in train_loader:
            loss = self._train_one_batch(dataloader, batch_indices, vector_store)
        return loss

    def _train_one_batch(self, dataloader, batch_indices, vector_store):
        # Get the actual data for these indices
        batch_images = torch.stack([dataloader[i][0] for i in batch_indices]).to(device)
        batch_labels = torch.tensor([dataloader[i][1] for i in batch_indices]).to(device)
        # Forward pass
        embeddings = self.forward(batch_images)
        # Compute loss
        loss = criterion(embeddings, batch_labels)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update FAISS index
        batch_indices_np = np.array(batch_indices)
        # Remove old vectors
        faiss_index.remove_ids(batch_indices_np)
        # Add updated vectors
        faiss_index.add_with_ids(embeddings.detach().cpu().numpy(), batch_indices_np)
        # Log with MLflow
        mlflow.log_metric("loss", loss.item())
        return loss