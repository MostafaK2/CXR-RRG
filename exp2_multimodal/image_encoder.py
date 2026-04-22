import torch
import torch.nn as nn
from torchvision import models


"""
        conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool x removed
        fc      x removed
"""

"""
output for transformer decoder: 
7x7 feature map          49 visual tokens
┌─┬─┬─┬─┬─┬─┬─┐         [v1,  v2,  v3,  ...  v49]
├─┼─┼─┼─┼─┼─┼─┤              ↓
├─┼─┼─┼─┼─┼─┼─┤         Transformer decoder
├─┼─┼─┼─┼─┼─┼─┤         cross-attends over these
├─┼─┼─┼─┼─┼─┼─┤         49 positions to generate
├─┼─┼─┼─┼─┼─┼─┤         each output word
├─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┘

"""

import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass


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
}


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