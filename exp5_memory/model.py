import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp2_multimodal.image_encoder import SwinEncoder
from exp2_multimodal.text_encoder  import ClinicalTextEncoder
from exp2_multimodal.fusion_module import CrossAttentionFusion
from exp2_multimodal.decoder import RRGDecoder


class DiseaseKnowledgeModule(nn.Module):
    def __init__(self, 
        d_model: int        = 512, 
        num_diseases: int   = 14, 
        topk: int           = 3,
        num_heads: int      = 8
        ):
        super().__init__()
        self.topk = topk

        self.num_diseases = num_diseases

        
        self.disease_emb = nn.Parameter(torch.empty(num_diseases, d_model))
        nn.init.xavier_uniform_(self.disease_emb.data)

        self.state_embedding = nn.Embedding(2, d_model) # state embedding (2, d_model)
        self.disease_pos_embedding = nn.Embedding(num_diseases, d_model) # positional embedding for (14, d_model)

        self.mlc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_diseases)
        )

        self.disease_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True
        )
        self.gate = nn.Linear(d_model, d_model)
    
    def forward(self, z_img, z_fused, labels=None):
        B      = z_fused.shape[0]
        device = z_fused.device

        # ── MLC prediction — linear head, unbounded logits ─────────────────
        pooled     = z_img.mean(dim=1)     # [B, d]
        mlc_logits = self.mlc_head(pooled)   # [B, D] — no compression issue

        # Disease position embeddings
        disease_idx = torch.arange(self.num_diseases, device=device)
        disease_idx = disease_idx.unsqueeze(0).expand(B, -1)  # [B, D]
        pos_emb     = self.disease_pos_embedding(disease_idx)  # [B, D, d]

        # ── State embeddings ────────────────────────────────────────────────
        if labels is not None:
            state_idx = labels.long()                              # [B, D] — ground truth
        else:
            state_idx = (torch.sigmoid(mlc_logits) > 0.5).long() # [B, D] — predicted

        state_emb = self.state_embedding(state_idx)               # [B, D, d]

        # ── Combine disease_emb + state + position ────────────────────────────
        K = self.disease_emb.unsqueeze(0) + state_emb + pos_emb    # [B, D, d]

        # Detach so gen loss cannot corrupt prototypes
        z_know, _ = self.disease_attn(z_fused, K.detach(), K.detach())

        gate  = torch.sigmoid(self.gate(z_fused))
        z_out = z_fused + gate * z_know

        return z_out, mlc_logits


class Multimodal_Memory_Real(nn.Module):
    def __init__(
            self, 
            d_model: int = 512,

            # Swin encoder
            img_enc_backbone: str = "swin_t",
            img_enc_freeze_layers: int = 8,  # FULLY FROZEN
                # FPN Configurations Dont change for SwinT
                use_fpn: bool = False,
                fpn_dim: int = 256,  
                fpn_scale: int = 2,

            # text encoder
            bert_model: str = "emilyalsentzer/Bio_ClinicalBERT", 
            bert_freeze_layers: int = 12,  # FULLY FROZEN
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
        self.image_encoder = SwinEncoder(
            backbone = img_enc_backbone,
            d_model = d_model,
            dropout = dropout,
            pretrained = True,
            freeze_layers = img_enc_freeze_layers,

            # FPN Configurations Dont change
            use_fpn = use_fpn,
            fpn_dim = fpn_dim,
            fpn_scale = fpn_scale
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

        self.knowledge = DiseaseKnowledgeModule(
            d_model=d_model,
            num_diseases=14,
            topk=5,
            num_heads=fusion_heads
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
    def forward(self, img, text, caption_ids, labels=None):

        # Pass though img and text encoders
        device = img.device
        image_tokens = self.image_encoder(img)               # [B, K, d_model]
        text_tokens, text_tok_mask = self.text_encoder(text, device) # [B, N, d_model], [B, N] Mask

        # Fuse img and text (cross attention) with res connection + gating
        z, _ = self.fusion(image_tokens, text_tokens, text_tok_mask) # [B, 49, d_model]

        z, mlc_logits =  self.knowledge(image_tokens, z, labels)

        # decoder outputs
        reports = self.decoder(z, caption_ids)
        return reports, mlc_logits


if __name__ == "__main__":
    model = Multimodal_Memory_Real(
        d_model=512,

        img_enc_backbone="swin_s",
        img_enc_freeze_layers=8,
        

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

    

