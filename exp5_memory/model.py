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
    def __init__(self, d_model, num_diseases, num_states=2, num_heads=8, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.num_diseases = num_diseases
        self.num_states = num_states  # k=2 (absent, present)

        # Memory matrix — THIS is both classifier and knowledge store
        # Shape: [num_diseases, num_states, d_model]
        self.M = nn.Parameter(
            torch.empty(num_diseases, num_states, d_model)
        )
        nn.init.xavier_uniform_(self.M)

        # Keep your disease attention for z enrichment
        self.disease_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True
        )

    def forward(self, z_fused, labels=None):
        B, S, d = z_fused.shape

        # Pool fused features — on Z not z_img
        z_pooled = z_fused.mean(dim=1)          # [B, d]

        # ── MLC via memory similarity (Li et al. Eq. 6) ──────────────────
        # M_flat: [num_diseases * num_states, d]
        M_flat = self.M.view(-1, d)

        # Similarity scores
        scores = z_pooled @ M_flat.T / (d ** 0.5)          # [B, num_diseases * num_states]
        scores = scores.view(B, self.num_diseases, self.num_states)

        # Softmax over states (present/absent) per disease
        alpha = torch.softmax(scores, dim=-1)               # [B, num_diseases, num_states]

        # MLC logits — probability of "present" state (index 1)
        mlc_probs = alpha[:, :, 1]                          # [B, num_diseases]

        # ── Classification loss (Li et al. Eq. 7) ────────────────────────
        # Returns raw probs for loss computation outside
        # Loss = -(1/n) Σ y_ij * log(alpha_ij)

        # ── Threshold for knowledge injection (Li et al.) ─────────────────
        alpha_hat = (mlc_probs > self.threshold).float()    # [B, num_diseases] binary

        # Knowledge embedding R = Σ α̂_ij · m_i (present state vectors)
        M_present = self.M[:, 1, :]                         # [num_diseases, d]
        R = alpha_hat @ M_present                           # [B, d]

        # Enrich z_fused with knowledge
        R_expanded = R.unsqueeze(1).expand(-1, S, -1)       # [B, S, d]
        z_out = z_fused + R_expanded                        # [B, S, d]

        return z_out, mlc_probs   # mlc_probs used for BCE loss outside

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
            num_states = 2,
            d_model=d_model,
            num_diseases=14,
            num_heads=fusion_heads, 
            threshold=0.2,
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

        z, mlc_logits =  self.knowledge(z, labels)

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

    

