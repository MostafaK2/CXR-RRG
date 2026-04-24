import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass

from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights
from typing import Dict, List, Tuple

@dataclass
class BackboneConfig:
    model_fn:     callable
    weights:      object
    out_channels: int
    num_layers:   int

@dataclass
class SwinConfig:
    """Configuration for Swin Transformer variants."""
    model_fn: callable
    weights: object
    embed_dim: int  # 96 for Swin-T
    depths: List[int]  # [2, 2, 6, 2] for Swin-T
    num_heads: List[int]  # [3, 6, 12, 24] for Swin-T
    num_layers: int  # Total number of block groups
    

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

SWIN_REGISTRY = {
    "swin_t": SwinConfig(model_fn=models.swin_t, weights=models.Swin_T_Weights.DEFAULT, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], num_layers=4,),
    "swin_s": SwinConfig(model_fn=models.swin_s, weights=models.Swin_S_Weights.DEFAULT, embed_dim=96, depths=[2, 2, 18, 2],num_heads=[3, 6, 12, 24], num_layers=4,),
    "swin_b": SwinConfig(model_fn=models.swin_b, weights=models.Swin_B_Weights.DEFAULT, embed_dim=128,depths=[2, 2, 18, 2],num_heads=[4, 8, 16, 32], num_layers=4,),
}


import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, 
                scales: int = 2, 
                fpn_dim: int = 256, 
                d_model: int = 512,
                c_per_stage: dict = None):
        super().__init__()
        assert 2 <= scales <= 4, "scales must be 2, 3, or 4"

        print(c_per_stage)
        
        self.laterals = nn.ModuleDict({
            str(i):  nn.Linear(c, fpn_dim) for i, c in c_per_stage.items()
        })

        self.output_proj = nn.Linear(fpn_dim, d_model)

    def forward(self, features: dict) -> torch.Tensor:
        """
            features: {stage_index: tensor (B, H, W, C)}
        """
        B = next(iter(features.values())).size(0)

        # 1. Project features into FPN Dimension (e.g. 256)
        projected = {}
        for idx, f in features.items():
            projected[idx] = self.laterals[str(idx)](f.reshape(B, -1, f.size(-1)))

        
        # Top-down fusion: coarsest → finest
        indices = sorted(projected.keys(), reverse=True)  # [7, 5, 3, 1]
        fused = {indices[0]: projected[indices[0]]}
        


        for i in range(1, len(indices)):
            coarse = fused[indices[i - 1]] # e.g. [2, 49, 256] (after lateral projection)
            fine   = projected[indices[i]] # e.g. [2, 192, 256] (after lateral projection)

            # 1. Calculate the spatial dimensions
            h_c = w_c = int(coarse.size(1) ** 0.5) # 7
            h_f = w_f = int(fine.size(1) ** 0.5)  # 14
            c = coarse.size(-1)

            # 2. Reshape to Spatial and Permute for PyTorch Ops: [B, L, C] -> [B, C, H, W]
            coarse_spatial = coarse.view(B, h_c, w_c, c).permute(0, 3, 1, 2).contiguous()

            coarse_up = F.interpolate(
                coarse_spatial, 
                size=(h_f, w_f), 
                mode="bilinear", 
                align_corners=False
            )
            
            coarse_up = coarse_up.permute(0, 2, 3, 1).reshape(B, h_f * w_f, c)
            fused[indices[i]] = fine + coarse_up

        # Project to d_model and concatenate finest → coarsest
        target_idx = indices[-1]
        z_vector = self.output_proj(fused[target_idx])
        return z_vector



# 0 Nothing(Full Training)
# 2 Stem + Stage 1.   56 x 56 (Texture/Edges)
# 4 Above + Stage 2   28 x 28 (Local Shapes)
# 6 Above + Stage 3   14 x 14 (Anatomy/Organs)
# 8 Entire Backbone   7  x  7 (Feature Extraction Only)

class SwinEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = "swin_s",
        d_model: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_layers: int = 8,

        # FPN Configurations Dont change
        use_fpn: bool = True,
        fpn_dim: int = 256,
        fpn_scale: int = 2
    ):

        super().__init__()
        assert backbone in SWIN_REGISTRY, (f"Unknown backbone '{backbone}'. "f"Available: {list(SWIN_REGISTRY.keys())}")
        assert freeze_layers 
        cfg = SWIN_REGISTRY[backbone]
        weights = cfg.weights if pretrained else None

        # 1. Load the backbone
        full_model = cfg.model_fn(weights=weights)
        self.backbone = full_model.features # This contains the stages

        # 2. Freezing the backbone 
        num_children = len(list(self.backbone.children()))
        assert 0 <= freeze_layers <= num_children, (
            f"freeze_layers ({freeze_layers}) must be between 0 and {num_children}."
        )
        self._freeze_backbone(freeze_layers)

        # ------------------------ 
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        c_per_stage = self._find_dimensions_per_stage()
        
        # FPN BASED
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FPN(scales=fpn_scale, fpn_dim=fpn_dim, d_model=d_model, c_per_stage=c_per_stage)
        else: 
            self.proj = nn.Linear(c_per_stage[7], d_model)
        
    
    def _find_dimensions_per_stage(self):
        STAGE_INDICES = [1, 3, 5, 7] 
        c_per_stage = {}
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            for i, layer in enumerate(self.backbone.children()):
                dummy = layer(dummy)
                if i in STAGE_INDICES:
                    c_per_stage[i] = dummy.size(-1)
        return c_per_stage

    def _freeze_backbone(self, freeze_layers):
        # child 0-7
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_layers:
                # Disable gradient calculation
                for param in child.parameters():
                    param.requires_grad = False
                
                # Switch to eval mode to lock LayerNorm statistics
                child.eval()
                print(f"Layer {i} frozen and set to eval().")
    
    def forward(self, x):
        
        feature_to_out = {}
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i == 5:
                feature_to_out[i] = x  
            if i == 7:
                feature_to_out[i] = x
                feat_s4 = x

        if self.use_fpn:
            return self.fpn(feature_to_out)
        
        out = feat_s4.reshape(feat_s4.size(0), -1, feat_s4.size(-1)) # [B, 49, 768]
        return self.proj(out)



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


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    encoder = SwinEncoder(freeze_layers=6, use_fpn=True)

    #  # ── Parameter summary ─────────────────────────────────────────────────
    trainable     = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    total         = trainable + non_trainable

    print(f"\n── Parameter Summary ──")
    print(f"  Trainable:     {trainable:,}")
    print(f"  Non-trainable: {non_trainable:,}")
    print(f"  Total:         {total:,}")

    x   = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    print(out.shape)
   