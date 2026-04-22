import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp2_multimodal.image_encoder import CNNEncoder
from exp2_multimodal.text_encoder  import ClinicalTextEncoder
from exp2_multimodal.fusion_module import CrossAttentionFusion
from exp2_multimodal.decoder import RRGDecoder


class Multimodal_Memory(nn.Module):
    def __init__(
            self, 
            d_model: int = 512,
            # CNN encoder
            cnn_backbone: str = "resnet50",
            cnn_freeze_layers: int = 8,
            # text encoder
            bert_model: str = "emilyalsentzer/Bio_ClinicalBERT", 
            bert_freeze_layers: int = 6,
            bert_max_length: int = 128,
            # Fusion
            fusion_heads: int = 8,
            fusion_ff_dim: int = 768,


            # Decoder
            vocab_size:        int   = 10000, # will be set
            decoder_layers:    int   = 6,
            decoder_heads:     int   = 8,
            decoder_ff_dim:    int   = 1024,
            decoder_max_len:   int   = 256,
            pad_id:            int   = 0,

            # ── Regularisation ─────────────────────────────────────────────────────
            dropout:           float = 0.1,

           
            ):
        super().__init__()

        # ── Image encoder ──────────────────────────────────────────────────────
        # Produces: [B, 49, d_model]
        self.image_encoder = CNNEncoder(
            backbone      = cnn_backbone,
            d_model       = d_model,
            freeze_layers = cnn_freeze_layers,
            dropout       = dropout,
        )
        
        # ── Clinical text encoder ──────────────────────────────────────────────
        # Produces: [B, N, d_model],  [B, N] mask
        self.text_encoder = ClinicalTextEncoder(
            model_name    = bert_model,
            d_model       = d_model,
            freeze_layers = bert_freeze_layers,
            max_length    = bert_max_length,
            dropout       = dropout,
        )

        # ── Cross-attention fusion ─────────────────────────────────────────────
        # Pass 1: image attends to clinical text
        # Pass 2: fused Z attends back to original image tokens
        # Produces: Z [B, 49, d_model]
        self.fusion = CrossAttentionFusion(
            d_model   = d_model,
            num_heads = fusion_heads,
            ff_dim    = fusion_ff_dim,
            dropout   = dropout,
        )

        # Decoder to generate reports
        self.decoder = RRGDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            max_len=decoder_max_len,
            d_ff=decoder_ff_dim,
            dropout=dropout,
            pad_id=pad_id
        )

        

    # Image [B, 224, 224], Text [B, Strngs], Caption_ids [B, T] (teacher forcing)
    def forward(self, img, text, caption_ids):

        # Pass though img and text encoders
        device = img.device
        image_tokens = self.image_encoder(img)               # [B, 49, d_model]
        text_tokens, text_tok_mask = self.text_encoder(text, device) # [B, N, d_model], [B, N] Mask

        # Fuse img and text (cross attention) with res connection + gating
        z, _ = self.fusion(image_tokens, text_tokens, text_tok_mask) # [B, 49, d_model]


        # use z to do Disease logits and tokens for extra superiviosn (NOT IMPLEMENTED YET) 

        # decoder outputs
        reports = self.decoder(z, caption_ids)
        return reports


if __name__ == "__main__":
    model = Multimodal_Memory(
        d_model=512,

        cnn_backbone="resnet50",
        cnn_freeze_layers=8,

        bert_model="emilyalsentzer/Bio_ClinicalBERT",
        bert_freeze_layers=6,
        bert_max_length=128,

        fusion_heads=8,
        fusion_ff_dim=768,

        vocab_size=1000,
        decoder_layers = 6,
        decoder_heads = 8,
        decoder_ff_dim = 1024,
        decoder_max_len = 256,
        pad_id = 0,

        dropout = 0.1
    )

