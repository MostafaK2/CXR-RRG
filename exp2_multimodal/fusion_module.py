import torch
import torch.nn as nn



# Sequential Refinement (the filter and Focus Approach)
    # - pass 1: image toks queries text tokens, asks which part of the lcincal text match my visual patches
    # - pass 2: 
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


         # ── Feed-forward ───────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )


        # ── Layer norms (pre-norm style — more stable) ─────────────────────────
        self.norm_img1  = nn.LayerNorm(d_model)   # Before first cross-attn
        self.norm_clin  = nn.LayerNorm(d_model)   # Before first cross-attn

        self.norm_img2= nn.LayerNorm(d_model)   # Before second attention
        self.norm2= nn.LayerNorm(d_model)   # Before second attention
        
        self.norm_ff    = nn.LayerNorm(d_model)   # Before feed-forward
        self.dropout    = nn.Dropout(dropout)


    def forward(self, 
        img_tokens: torch.Tensor, # [B, 49, d_model],
        clincal_text_token: torch.Tensor, # [B, N, d_model], 
        clincal_mask: torch.Tensor # True = ignore pos
        ) -> torch.Tensor:



        # ------------ Attention Pass 1 ------------------
        # Seaches and creates textual informed 
        q1 = self.norm_img1(img_tokens)
        k1 = v1 = self.norm_clin(clincal_text_token)
        ca_out1, _ = self.cross_attn(
            query = q1, 
            key = k1,
            value = v1,
            key_padding_mask = clincal_mask
        )
        
        # textually informed visual tokens
        x = img_tokens + self.dropout(ca_out1)

        # ------------ Attention Pass 1 ------------------
        q2 = v2 = self.norm_img2(img_tokens)
        k2 = self.norm2(x)

        # given what I know clinically find what image tokens I should pay more attention too. (refied image feature) pulling i
        ca_out2, attn_weights = self.cross_attn2(
            query=q2,
            key = k2,
            value = v2
        )

        # residual conncetion (Combination of extually informed visual tokens + refined image features)
        x = x + self.dropout(ca_out2) # This essentially adds ca_out1 and ca_out2

        # --------- FFN ----------
        x = x + self.dropout(self.ffn(self.norm_ff(x)))


        return x, attn_weights # Z vector [B, 49, d_model]



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

    img_tokens  = torch.randn(B, N_img, 512).to(device)
    clin_tokens = torch.randn(B, N_txt, 512).to(device)

    # Simulate padding mask — last 5 tokens are padding for all samples
    clin_mask   = torch.zeros(B, N_txt, dtype=torch.bool).to(device)
    clin_mask[:, -4:] = True

    out, attn = fusion(img_tokens, clin_tokens, clin_mask)

    print(out.shape)
