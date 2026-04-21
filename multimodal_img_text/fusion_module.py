import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module): 
    def __init__(self, 
        d_model:   int   = 512,
        num_heads: int   = 8,
        ff_dim:    int   = 2048,
        dropout:   float = 0.1):

        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Image tokens (Q) attend over clinical tokens (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        
        # second attention pass
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # ── Soft gate ──────────────────────────────────────────────────────────
        # Conditioned on pooled clinical embedding
        # Empty string → [CLS][SEP] pool → gate ≈ 0
        # Real text    → rich pool       → gate ≈ 1
        self.gate_linear = nn.Linear(d_model, 1)

         # ── Feed-forward ───────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )


        # ── Layer norms (pre-norm style — more stable) ─────────────────────────
        self.norm_img   = nn.LayerNorm(d_model)   # applied to image tokens before cross-attn
        self.norm_clin  = nn.LayerNorm(d_model)   # applied to clinical tokens before cross-attn
        self.norm_ff    = nn.LayerNorm(d_model)   # applied before feed-forward

        self.dropout    = nn.Dropout(dropout)

    def _masked_pool(
        self,
        tokens:    torch.Tensor,   # [B, N, d_model]
        attn_mask: torch.Tensor,   # [B, N] bool — True = ignore
    ) -> torch.Tensor:             # [B, d_model]
        # attn_mask: True = ignore → invert to get real token positions
        real_mask   = (~attn_mask).float().unsqueeze(-1)    # [B, N, 1]  1=real, 0=pad
        pooled      = (tokens * real_mask).sum(dim=1)       # [B, d_model]
        n_real      = real_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]     avoid div/0
        return pooled / n_real  

    def forward(self, 
        img_tokens: torch.Tensor, # [B, 49, d_model],
        clincal_text_token: torch.Tensor, # [B, N, d_model], 
        clincal_mask: torch.Tensor # True = ignore pos
        ) -> torch.Tensor:

        # pooling clin text for gate
        pooled_clin = self._masked_pool(clincal_text_token, clincal_mask)
        gate = torch.sigmoid(self.gate_linear(pooled_clin))

        #pre norms
        norm_img = self.norm_img(img_tokens)    # [B, 49, d_model]
        norm_clin = self.norm_clin(clincal_text_token)


        # Image tokens (Q) and Text tokens (K, V): Saying how the image 
        # "what clinical history are relevant"
        cross_out, attn_weight = self.cross_attn(
            query = norm_img, 
            key = norm_clin,
            value = norm_clin,
        )

        # residual + gated
        Z = img_tokens + (gate.unsqueeze(1) * self.dropout(cross_out))

        # "given what I know clinically, look again at the image"
        cross_attn2, attn_weights = self.cross_attn2(
            query=Z,
            key = img_tokens,
            value = img_tokens
        )
        Z = Z + self.dropout(cross_attn2)

        Z = Z + self.ffn(self.norm_ff(Z))

        return Z, attn_weight # Z vector [B, 49, d_model]



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B        = 4
    N_img    = 49     # 7x7 from CNN encoder
    N_txt    = 6     # simulated token sequence length

    fusion = CrossAttentionFusion(
        d_model   = 512,
        num_heads = 8,
        ff_dim    = 728,
        dropout   = 0.1,
    ).to(device)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(total_params)

    img_tokens  = torch.randn(B, N_img, 512).to(device)
    clin_tokens = torch.randn(B, N_txt, 512).to(device)

    # Simulate padding mask — last 5 tokens are padding for all samples
    clin_mask   = torch.zeros(B, N_txt, dtype=torch.bool).to(device)
    clin_mask[:, -3:] = True

    out, attn = fusion(img_tokens, clin_tokens, clin_mask)

    print(out.shape)
