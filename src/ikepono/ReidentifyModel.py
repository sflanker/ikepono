import torch.nn as nn
import timm
import torch

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
        self.out_features = model_configuration["out_features"]
        self.dropout = model_configuration["dropout"]
        self.hidden_units = model_configuration["hidden_units"]
        self.device = torch.device(model_configuration["device"])

        self.backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained)
        if not self.pretrained:
            _init_weights(self.backbone)
        else:
            self.backbone.requires_grad_(not self.freeze)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:self.cut]).to(self.device)
        self.projection = self._create_head(self.backbone_output_dim, self.out_features).to(self.device)
        _init_weights(self.projection)

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        backbone_representation = self.backbone(img_tensor)
        embedding = self.projection(backbone_representation)

        return embedding

    def _create_head(self, backbone_output_dim, out_features):
         return nn.Sequential(
            nn.Linear(backbone_output_dim, backbone_output_dim),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(backbone_output_dim, self.hidden_units),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units, out_features)
        )