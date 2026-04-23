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


class MultiHeadKnowledgeMemory(nn.Module):
    def __init__(
            self,
            n_slots: int,
            n_labels: int,
            d_model: int, 
            n_heads: int = 4,
            dropout: float = 0.1
          
    ):
        super().__init__()

        self.n_slots = n_slots
        
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.2) # (N_slots, d_model) --> act as memory bank

        # Slots serve as query, and Key/Value serve as Z (fused tokens)
        # Output (N, d_model): each slots asks what in this image is relevant to them
        self.update_slots = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_labels)
        )

        self.fuse_mem = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )


    def forward(self, z):
        # Z = [B, 49, d_model]
        B = z.size(0)
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)     # [B, N_slots, d_model]. # Memory for all the abtches

        updated_slots, _ = self.update_slots(
            query = slots, # [B, N, d_model]
            key = z,       # [B, 49, D_model] Fused Feature
            value = z,     # [B, 49, D_model] Fused features
        ) #. -> [B, N, d_model]

        # for very image tokens what knowledge is relevant to me?
        z_enhanced, _ = self.fuse_mem(
            query = z,              # [B, 49, D]
            key   = updated_slots,  # [B, K, D]
            value = updated_slots,  # [B, K, D]
        ) # --> z_enhanced
        z_enhanced = z + z_enhanced  # residual Concatenated

        pooled_mem = updated_slots.mean(dim=1) # [B, D_dims]
        cls_logits   = self.cls_head(pooled_mem)             # [B, 14]

        return z_enhanced, updated_slots, cls_logits



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

            # Memory Augmentation
            n_slots: int = 100,
            n_labels: int = 14,
            memory_n_heads: int = 8,
            
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

        self.memory = MultiHeadKnowledgeMemory(
            n_slots = n_slots,
            n_labels = n_labels,
            d_model = d_model, 
            n_heads = memory_n_heads,
            dropout = dropout
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

        # Get enhanced z and cls_logits
        z_enhanced, updated_memory, cls_logits =  self.memory(z)

        # decoder outputs
        reports = self.decoder(z_enhanced, caption_ids)

        return reports, cls_logits, updated_memory


# if __name__ == "__main__":
#     model = Multimodal_Memory(
#         d_model=512,

#         cnn_backbone="densenet121",
#         cnn_freeze_layers=8,

#         bert_model="emilyalsentzer/Bio_ClinicalBERT",
#         bert_freeze_layers=6,
#         bert_max_length=128,

#         fusion_heads=8,
#         fusion_ff_dim=768,

#         vocab_size=1000,
#         decoder_layers = 6,
#         decoder_heads = 8,
#         decoder_ff_dim = 1024,
#         decoder_max_len = 256,
#         pad_id = 0,

#         n_slots =  100,
#         n_labels = 14,
#         memory_n_heads = 8,

#         dropout = 0.1
#     )
#     device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     caption_ids = torch.randint(0, 1000, (3, 1)).to(device)
#     images = torch.randn(3, 3, 224, 224).to(device)
#     texts = ["", "", "Heart size normal"]

#     reports, cls_logits, updated_mem = model(images, texts, caption_ids)
#     print(reports.shape)
#     print(cls_logits.shape)

