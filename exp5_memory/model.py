import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from exp2_multimodal.image_encoder import SwinEncoder
from exp2_multimodal.text_encoder  import ClinicalTextEncoder
from exp2_multimodal.fusion_module import CrossAttentionFusion
from exp2_multimodal.decoder import RRGDecoder

# Option B — Soft per-patch
#     + Spatially aware — each patch enriched by its own disease context
#     + Richer gradient signal
#     + Novel contribution beyond Li et al.
#     - Slightly more parameters in gradient graph
#     - Needs ablation to justify

class DiseaseKnowledgeModule(nn.Module):
    def __init__(self, d_model, num_diseases, num_states=2, num_heads=8, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.num_diseases = num_diseases
        self.num_states = num_states  # k=2 (absent, present)

        # Memory matrix — THIS is both classifier and knowledge store
        # Shape: [num_diseases, num_states, d_model]
        self.disease_knowledge = nn.Parameter(
            torch.empty(num_diseases, num_states, d_model)
        )
        nn.init.xavier_uniform_(self.disease_knowledge) # intialize Memory Matrix

        # # Keep your disease attention for z enrichment
        # self.disease_attn = nn.MultiheadAttention(
        #     d_model, num_heads=num_heads, batch_first=True
        # )

        self.gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),  # [z_fused || R_per_patch] → gate
            nn.Sigmoid()
        )

    def forward(self, z_fused, labels=None):
        B, S, d_model = z_fused.shape
        

        query = z_fused
        key = self.disease_knowledge

        # print(self.disease_knowledge.shape)
        flat_memory= self.disease_knowledge.view(-1, d_model)
        
        flat_memory = flat_memory.unsqueeze(0).repeat(B,1,1)
        # print("flat memory repeat: ", flat_memory.shape)

        score_per_patch = (query @ flat_memory.transpose(-1, -2)) / math.sqrt(d_model)
       #  print("score per patch: ", score_per_patch.shape)

        # Attention weighted pooling maybe (IMPLEMENT LAtER)
        average_score_perimg = score_per_patch.mean(dim=1) 
        # print("average scoring per: ", average_score_perimg.shape)

        scores_pooled = average_score_perimg.view(B, self.num_diseases, self.num_states)
        # print("scores_pooled: ",scores_pooled.shape)
        
        # ------------- SIMILARITY SCORING AND PATCH SCORING COMPLETE Z_FUSED AND DISESE EMB
        alpha_overall = torch.softmax(scores_pooled, dim=-1) 
        alpha_per_patch = torch.softmax(score_per_patch.view(B, S, self.num_diseases, self.num_states), dim=-1) 
        # print("  pooled alpha: ",alpha_overall.shape)
        # print("  per patch alpha: ", alpha_per_patch.shape)
        # print("-------------------- Per Patch Memory Retrieval ----------------- \n")
        
        # For every patch select only present (1) for the 14 diseases
        mlc_probs_per_patch = alpha_per_patch[:, :, :, 1]; # print("  mlc_prob_per_patch: " , mlc_probs_per_patch.shape)  # (B, S, D)
        mlc_probs, _ = mlc_probs_per_patch.max(dim=1); # print("  mlc props: " , mlc_probs.shape) # (B, diseases) 14 disesae pred for each image (B)
        M_present = self.disease_knowledge[:, 1, :]; # print("  Memory present: ", M_present.shape) # [Disease, dimension] # selcts mem emb for each disease (present)

        R_per_patch = torch.einsum(
            'bsi, id -> bsd',
            mlc_probs_per_patch,    # [B, S, 14] — per patch present probs
            M_present               # [14, 512]
        )      # knowledge summary vector
        
        # print("R per Patch: ", R_per_patch.shape)
   
        
        # m_{b,s,d} = \sum_{k=1}^{K} \alpha_{b,s,d,k} \, M_{d,k} -> m_{b,s,d} \in \mathbb{R}^{d}
        # In forward:
        gate_input = torch.cat([z_fused, R_per_patch], dim=-1)  # [B, S, 1024]
        gate        = self.gate(gate_input) 
        z_out = z_fused + gate * R_per_patch   # knowledge-enriched fused features
        
        return z_out, mlc_probs
    











        # B, S, d = z_fused.shape

        # # Pool fused features — on Z not z_img
        # z_pooled = z_fused.mean(dim=1)          # [B, d]

        # # ── MLC via memory similarity (Li et al. Eq. 6) ──────────────────
        # # M_flat: [num_diseases * num_states, d]
        # M_flat = self.M.view(-1, d)

        # # Similarity scores
        # scores = z_pooled @ M_flat.T / (d ** 0.5)          # [B, num_diseases * num_states]
        # scores = scores.view(B, self.num_diseases, self.num_states)

        # # Softmax over states (present/absent) per disease
        # alpha = torch.softmax(scores, dim=-1)               # [B, num_diseases, num_states]

        # # MLC logits — probability of "present" state (index 1)
        # mlc_probs = alpha[:, :, 1]                          # [B, num_diseases]

        # # ── Classification loss (Li et al. Eq. 7) ────────────────────────
        # # Returns raw probs for loss computation outside
        # # Loss = -(1/n) Σ y_ij * log(alpha_ij)

        # # ── Threshold for knowledge injection (Li et al.) ─────────────────
        # alpha_hat = (mlc_probs > self.threshold).float()    # [B, num_diseases] binary

        # # Knowledge embedding R = Σ α̂_ij · m_i (present state vectors)
        # M_present = self.M[:, 1, :]                         # [num_diseases, d]
        # R = alpha_hat @ M_present                           # [B, d]

        # # Enrich z_fused with knowledge
        # R_expanded = R.unsqueeze(1).expand(-1, S, -1)       # [B, S, d]
        # z_out = z_fused + R_expanded                        # [B, S, d]

        # return z_out, mlc_probs   # mlc_probs used for BCE loss outside

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
    B =3
    d_model = 512
    S = 49
    model =  DiseaseKnowledgeModule(d_model=512, num_diseases=14, num_states=2).to("cuda")
    x = torch.randn(B, S, d_model).to("cuda")
    print("input shpae: ", x.shape)
    model(x)

    # model = Multimodal_Memory_Real(
    #     d_model=512,

    #     img_enc_backbone="swin_s",
    #     img_enc_freeze_layers=8,
        

    #     bert_model="emilyalsentzer/Bio_ClinicalBERT",
    #     bert_freeze_layers=6,
    #     bert_max_length=128,

    #     fusion_heads=8,
    #     fusion_ff_dim=768,

    #     vocab_size=1000,
    #     decoder_layers = 6,
    #     decoder_heads = 8,
    #     decoder_ff_dim = 1024,
    #     decoder_max_len = 256,
    #     pad_id = 0,

    #     dropout = 0.1
    # )

    

