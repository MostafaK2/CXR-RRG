print(__file__)

# model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp2_multimodal.image_encoder import SwinEncoder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch.nn as nn


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
        img_enc_backbone: str    = "swin_s",
        img_enc_dim: int         = 512,
        img_enc_freeze_layer: int = 8,

        dropout: float           = 0.1,
        max_new_tokens: int      = 256,

        # LLM Configs
        model_name: str          = "Qwen/Qwen2.5-3B",

        # How many LLM layers to unfreeze from the top
        unfreeze_last_n_layers: int = 4,
    ):
        super().__init__()

        # ── Image Encoder ──────────────────────────────────────────────────────
        self.image_encoder = SwinEncoder(
            backbone=img_enc_backbone,
            d_model=img_enc_dim,
            dropout=dropout,
            freeze_layers=img_enc_freeze_layer,
        )

        # ── Tokenizer ──────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ── LLM — load in bfloat16 explicitly ─────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,   # ← explicit, avoids ambiguity
            use_safetensors=True,
        )

        # Freeze everything first
        for p in self.llm.parameters():
            p.requires_grad = False


        # Always unfreeze LM head — needed to predict report tokens
        # for p in self.llm.lm_head.parameters():
        #     p.requires_grad = True

        # ── Visual Alignment — match LLM dtype immediately ────────────────────
        d_llm = self.llm.config.hidden_size
        self.visual_alignment = LinearAlignment(img_enc_dim, d_llm, dropout).to(torch.bfloat16)

        self.max_new_tokens = max_new_tokens
        self.prompt = "Write a detailed chest xray report in standard clinical style from the given inputs."

    # ── dtype helper ───────────────────────────────────────────────────────────
    def _cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast any tensor to the LLM's dtype (bfloat16)."""
        return tensor.to(self.llm.dtype)

    # ── Prompt embeddings ──────────────────────────────────────────────────────
    def _prompt_embeds(self, B: int, device: torch.device):
        tok = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=30,
        ).to(device)

        # NOTE: no torch.no_grad() here — keep in graph so gradients can flow
        print(next(model.visual_alignment.parameters()).dtype)

        embeds = self.llm.get_input_embeddings()(tok.input_ids)   # [1, P, d_llm]
        embeds = self._cast(embeds)
        embeds = embeds.expand(B, -1, -1).contiguous()            # [B, P, d_llm]
        mask   = torch.ones(B, tok.input_ids.size(1), dtype=torch.long, device=device)
        return embeds, mask

    # ── Clinical text embeddings ───────────────────────────────────────────────
    def _text_embeds(self, texts, device: torch.device):
        cleaned = [t if t.strip() else "[No indication]" for t in texts]
        tok = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=40,
        ).to(device)

        # NOTE: no torch.no_grad() — keep in graph
        embeds = self.llm.get_input_embeddings()(tok.input_ids)   # [B, T, d_llm]
        embeds = self._cast(embeds)
        return embeds, tok.attention_mask

    # ── Forward (training) ────────────────────────────────────────────────────
    def forward(self, images, texts, report_ids, report_mask):
        B, device = images.size(0), images.device

        # Visual tokens → align → cast to bfloat16
        visual_tokens         = self._cast(self.image_encoder(images))                 # [B, N, d_vis]
        aligned_visual_tokens = self._cast(self.visual_alignment(visual_tokens))  # [B, N, d_llm]
        vis_mask              = torch.ones(B, aligned_visual_tokens.size(1),
                                           dtype=torch.long, device=device)

        # Clinical text embeddings
        txt_embeds, txt_mask       = self._text_embeds(texts, device)      # [B, T, d_llm]

        # Prompt embeddings
        prompt_embeds, prompt_mask = self._prompt_embeds(B, device)        # [B, P, d_llm]

        # Report embeddings (teacher forcing targets)
        report_embeds = self._cast(
            self.llm.get_input_embeddings()(report_ids)                    # [B, R, d_llm]
        )

        # Concatenate full sequence: [visual | clinical | prompt | report]
        inputs_embeds  = torch.cat(
            [aligned_visual_tokens, txt_embeds, prompt_embeds, report_embeds], dim=1
        )
        attention_mask = torch.cat(
            [vis_mask, txt_mask, prompt_mask, report_mask], dim=1
        )

        # Labels: ignore prefix, train only on report tokens
        prefix_len = (
            aligned_visual_tokens.size(1)
            + txt_embeds.size(1)
            + prompt_embeds.size(1)
        )
        ignore = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore, report_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100  # mask padding

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ── Generate (inference) ──────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, images, texts, num_beams: int = 4):
        device = images.device
        B      = images.size(0)

        # Visual tokens
        visual_tokens         = self.image_encoder(images)
        aligned_visual_tokens = self._cast(self.visual_alignment(visual_tokens))
        vis_mask              = torch.ones(B, aligned_visual_tokens.size(1),
                                           dtype=torch.long, device=device)

        # Clinical text
        txt_embeds, txt_mask       = self._text_embeds(texts, device)

        # Prompt
        prompt_embeds, prompt_mask = self._prompt_embeds(B, device)

        # Prefix only — no report tokens at inference
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


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = Radiology_llm().to("cuda")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")

    # Dtype check
    dummy_ids = torch.tensor([[1, 2, 3]]).cuda()
    txt_emb   = model.llm.get_input_embeddings()(dummy_ids)
    vis_tok   = model.image_encoder(torch.randn(1, 3, 224, 224).cuda())
    aln_tok   = model.visual_alignment(vis_tok)

    print(f"LLM dtype:             {model.llm.dtype}")
    print(f"Text embeddings dtype: {txt_emb.dtype}")
    print(f"Visual tokens dtype:   {aln_tok.dtype}")
    # All three should print: torch.bfloat16

    B          = 2
    images     = torch.randn(B, 3, 224, 224).cuda()
    texts      = ["Shortness of breath, rule out pneumonia", ""]
    report_ids = torch.randint(1, 1000, (B, 64)).cuda()
    report_msk = torch.ones(B, 64, dtype=torch.long).cuda()

    out = model(images, texts, report_ids, report_msk)
    print(f"Loss: {out.loss}")

    with torch.no_grad():
        generated = model.generate(images, texts)
        print(f"Generated: {generated}")