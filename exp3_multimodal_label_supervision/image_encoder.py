import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass

from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights


@dataclass
class BackboneConfig:
    model_fn:     callable
    weights:      object
    out_channels: int
    num_layers:   int


BACKBONE_REGISTRY = {
    # ── ResNet family ─────────────────────────────────────────────────────────
    "resnet18":  BackboneConfig(models.resnet18,  models.ResNet18_Weights.DEFAULT,  512,  8),
    "resnet34":  BackboneConfig(models.resnet34,  models.ResNet34_Weights.DEFAULT,  512,  8),
    "resnet50":  BackboneConfig(models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048, 8),
    "resnet101": BackboneConfig(models.resnet101, models.ResNet101_Weights.DEFAULT, 2048, 8),
    # ── DenseNet family ───────────────────────────────────────────────────────
    "densenet121": BackboneConfig(models.densenet121, models.DenseNet121_Weights.DEFAULT, 1024, 12),
    "densenet169": BackboneConfig(models.densenet169, models.DenseNet169_Weights.DEFAULT, 1664, 12),
    "densenet201": BackboneConfig(models.densenet201, models.DenseNet201_Weights.DEFAULT, 1920, 12),

    "swin_t": BackboneConfig(models.swin_t, models.Swin_T_Weights.DEFAULT, 768, 12),
}

class SwinEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = "swin_t",
        d_model: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):

        super().__init__()

        assert backbone == "swin_t", "Add more variants if needed"

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = swin_t(weights=weights)

        # ── remove classification head ──
        self.backbone = model.features   # <-- key part
        # Swin output dim
        embed_dim = 768
        self.projection = nn.Linear(embed_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.backbone(x)  # (B, num_tokens, C)
        # Swin sometimes returns (B, H, W, C)
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(
        self,
        backbone:      str   = "resnet18",   # ← just change this
        d_model:       int   = 512,
        dropout:       float = 0.1,
        freeze_layers: int   = 8,
        pretrained:    bool  = True,
    ):
        super().__init__()

        assert backbone in BACKBONE_REGISTRY, (f"Unknown backbone '{backbone}'.\n" f"Available: {list(BACKBONE_REGISTRY.keys())}")

        cfg     = BACKBONE_REGISTRY[backbone]
        weights = cfg.weights if pretrained else None

        assert 0 <= freeze_layers <= cfg.num_layers, (
            f"freeze_layers must be between 0 and {cfg.num_layers} for {backbone}"
        )

        model = cfg.model_fn(weights=weights)

         # ── Setup backbone ────────────────────────────────────────────────────
        if "densenet" in backbone:
            self.backbone = model.features
            self.pool     = nn.AdaptiveAvgPool2d(7)
        else:
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.pool     = nn.Identity()

        # ── Projection ────────────────────────────────────────────────────────
        self.projection = nn.Linear(cfg.out_channels, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

        # ── Freeze Backbone ────────────────────────────────────────────────────────
        self._freeze_backbone(freeze_layers)

    def _freeze_backbone(self, freeze_layers):
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.backbone(images))          # (B, C, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)      # (B, 49, C)
        x = self.dropout(self.norm(self.projection(x)))  # (B, 49, d_model)
        return x


# # ── Usage ─────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     encoder = CNNEncoder(backbone="densenet121", d_model=512, freeze_layers=12)

#      # ── Parameter summary ─────────────────────────────────────────────────
#     trainable     = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
#     non_trainable = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
#     total         = trainable + non_trainable

#     print(f"\n── Parameter Summary ──")
#     print(f"  Trainable:     {trainable:,}")
#     print(f"  Non-trainable: {non_trainable:,}")
#     print(f"  Total:         {total:,}")

#     x   = torch.randn(2, 3, 224, 224)
#     out = encoder(x)
#     print(out.shape)  # (2, 49, 512)