import torch
import torch.nn as nn
import torch.nn.functional as F


class UAVEmbedderMultiTask(nn.Module):
    """Multi-task DINOv2 ViT-B/14 embedder for global retrieval and yaw estimation."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        for param in self.backbone.parameters():
            param.requires_grad = False

        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        self.backbone.norm.requires_grad_(True)

        self.embed_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        embeddings = F.normalize(self.embed_head(features), p=2, dim=1)
        rotation_sincos = F.normalize(self.rotation_head(features), p=2, dim=1)
        return embeddings, rotation_sincos