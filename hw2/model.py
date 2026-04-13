"""Network architectures: modified ResNet-18, projector head, and combined models."""

import torch
import torch.nn as nn
import torchvision.models as models

from config import Config


# ---------- Backbone ----------

def get_backbone(config: Config) -> nn.Module:
    """Create a modified ResNet-18 suitable for CIFAR-10 (32x32 inputs).

    Modifications from the standard torchvision ResNet-18:
      1. conv1: 3x3, stride=1, padding=1 (was 7x7, stride=2, padding=3)
      2. maxpool after conv1 replaced with nn.Identity()
      3. Final fc layer replaced with nn.Identity() -> output is 512-dim
    All weights are randomly initialised (no pretrained weights).
    """
    backbone = models.resnet18(weights=None)
    # Smaller first convolution for 32x32 images
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove aggressive downsampling
    backbone.maxpool = nn.Identity()
    # Remove classification head; keep avgpool + flatten -> 512-dim
    backbone.fc = nn.Identity()
    return backbone


# ---------- Projector Head ----------

class ProjectorHead(nn.Module):
    """Two-layer MLP projector: input_dim -> hidden_dim -> output_dim."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------- Combined Models ----------

class SimCLRModel(nn.Module):
    """Backbone + projector for SimCLR self-supervised training.

    forward() returns (features, projections):
      - features:    backbone output (512-dim), used for kNN monitor
      - projections: projector output (128-dim), used for NT-Xent loss
    """

    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)
        if config.use_projector:
            self.projector = ProjectorHead(
                config.feature_dim,
                config.projector_hidden_dim,
                config.projector_out_dim,
            )
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)           # (B, 512)
        projections = self.projector(features) # (B, 128) or (B, 512)
        return features, projections


class SupervisedModel(nn.Module):
    """Backbone + linear classification head for supervised training."""

    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class LinearClassifier(nn.Module):
    """Single linear layer for linear probing evaluation."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
