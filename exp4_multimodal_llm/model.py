# model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp2_multimodal.image_encoder import SwinEncoder   # your file
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


class LinearAlignment(nn.Module):
    def __init__(self, d_visual: int, d_llm: int, dropout: float):
        super().__init__()
        
        d_mid = (d_visual + d_llm) // 2
        self.net = nn.Sequential(
            nn.Linear(d_visual, d_mid),
            nn.LayerNorm(d_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, d_llm),
            nn.LayerNorm(d_llm),
        )

    def forward(self, x):
        return self.net(x)

class Radiology_llm(nn.Module):
    def __init__(
        self,
        img_enc_backbone: str      = "swin_s",
        img_enc_dim: int           = 512,
        img_enc_freeze_layer: int   = 8,
        dropout: float          = 0.1,
        max_new_tokens: int     = 256,

        # LLM Configs
        model_name: str         = "stanford-crfm/BioMedLM" 
    ):
        super().__init__()
 
        # Image Encoder
        self.image_encoder = SwinEncoder(
            backbone=img_enc_backbone,
            d_model=img_enc_dim,
            dropout=dropout,
            freeze_layers=img_enc_freeze_layer,
        )

        # Tokenizer for llm
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # LLM decoder model Frozen
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        d_llm = self.llm.config.hidden_size
 
        self.visual_alignment = LinearAlignment(img_enc_dim, d_llm, dropout)
        self.max_new_tokens = max_new_tokens

        self.prompt = "Write a detailed chest x-ray report in standard clinical style from the given inputs."
    
    def _prompt_embeds(self, B, device):
        tok = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=30,
        ).to(device)

        with torch.no_grad():
            embeds = self.llm.get_input_embeddings()(tok.input_ids)  # [1, P, d_llm]

        embeds = embeds.expand(B, -1, -1).contiguous()               # [B, P, d_llm]
        mask   = torch.ones(B, tok.input_ids.size(1), dtype=torch.long, device=device)
        return embeds, mask
    
    
    def _text_embeds(self, texts, device):
        cleaned = [t if t.strip() else "[No indication]" for t in texts]
        tok = self.tokenizer(
            cleaned, return_tensors="pt",
            padding=True, truncation=True, max_length=40
        ).to(device)
        with torch.no_grad():
            embeds = self.llm.get_input_embeddings()(tok.input_ids)
        return embeds, tok.attention_mask
 
    
    def forward(self, images, texts, report_ids, report_mask):
        B, device= images.size(0), images.device

        # Image Encoder -> Visual Alignment to LLM Dim
        visual_tokens  = self.image_encoder(images)                                # [B, N, d_visual]
        aligned_visual_tokens  = self.visual_alignment(visual_tokens)              # [B, N, d_llm]

        # [B, 49] — all ones, telling BioGPT to attend to all visual tokens
        vis_mask = torch.ones(B, aligned_visual_tokens.size(1), dtype=torch.long, device=device)   
        
        # Clinical Text Embeddings
        txt_embeds, txt_mask = self._text_embeds(texts, device)  # [B, T_text, 1024], [B, T_text]

        # prompt embeddings 
        prompt_embeds, prompt_mask = self._prompt_embeds(B, device)


        # TARGET REPORT EMBEDDINGS
        report_embeds = self.llm.get_input_embeddings()(report_ids) # [B, Report_T, 1024]
        
        # Concatenate [B, visual + txt + prompt + report, 1024]
        inputs_embeds  = torch.cat([aligned_visual_tokens, txt_embeds, prompt_embeds, report_embeds], dim=1)
        # ---> total input size [49 + 40 + 40 + 256]

        # [B, 196 + T_text + T_report]  — 1 everywhere except padding
        attention_mask = torch.cat([vis_mask,  txt_mask,  prompt_mask, report_mask],   dim=1) 

        # ignore (fill -100) visual + text when calculating loss
        prefix_len = aligned_visual_tokens.size(1) + txt_embeds.size(1) + prompt_embeds.size(1)
        ignore  = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
        labels  = torch.cat([ignore, report_ids], dim=1) 
        labels[labels == self.tokenizer.pad_token_id] = -100 # Also ignore pad tokens for loss calculation
 
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
 
    @torch.no_grad()
    def generate(self, images, texts, num_beams: int = 4):
        device = images.device
        B      = images.size(0)
 
        visual_tokens         = self.image_encoder(images)
        aligned_visual_tokens = self.visual_alignment(visual_tokens)
        vis_mask              = torch.ones(B, aligned_visual_tokens.size(1), dtype=torch.long, device=device)
 
        txt_embeds, txt_mask  = self._text_embeds(texts, device)
        prompt_embeds, prompt_mask = self._prompt_embeds(B, device)

 
        inputs_embeds  = torch.cat([aligned_visual_tokens, txt_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([vis_mask, txt_mask, prompt_mask], dim=1)
 
        ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

 
 
if __name__ == "__main__":
    model  = Radiology_llm().to("cuda")
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")
 
    B = 2
    images     = torch.randn(B, 3, 224, 224).cuda()

    texts      = ["Shortness of breath, rule out pneumonia", ""]
    report_ids = torch.randint(1, 1000, (B, 64)).cuda()
    report_msk = torch.ones(B, 64, dtype=torch.long).cuda()
 
    out = model(images, texts, report_ids, report_msk)
    print(out.loss)
    with torch.no_grad():
        model.generate(images, texts)
    

    