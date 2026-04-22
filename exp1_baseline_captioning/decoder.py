import torch
import torch.nn as nn
from encoder import CNNEncoder

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: (B,T,d_model)
        """
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1,T)
        return x + self.pos(positions)

class ChestXrayReportGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 max_len=64, d_ff=128, dropout=0.1, pad_id=0, freeze_enc_layers=8):
        super().__init__()
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = LearnablePositionalEmbedding(max_len=max_len, d_model=d_model)
        self.img_enc = CNNEncoder(d_model=d_model, dropout=dropout, freeze_layers=freeze_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True  # norm_first=True = Pre-LN, like yours
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        
        # decoder_out -> vocab
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    
    def forward(self, images, caption_ids):
        B, T = caption_ids.shape
        img_tokens = self.img_enc(images)  # (B, N_img, D)

        # Embeddings
        x = self.emb(caption_ids) # captions
        x = self.pos(x)           #  positional embedding

        # Masks
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=caption_ids.device).bool()
        pad_mask = (caption_ids == self.pad_id)  # (B, T) — PyTorch expects this shape here
        
        # Decode
        out = self.decoder(
            tgt=x,
            memory=img_tokens,
            tgt_mask=causal_mask,           # (T, T) causal
            tgt_key_padding_mask=pad_mask,  # (B, T) padding
        )

        logits = self.head(self.ln_f(out))
        return logits
    