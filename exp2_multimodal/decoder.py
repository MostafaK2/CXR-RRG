import torch
import torch.nn as nn

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
    
class RRGDecoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 n_layers: int = 6,
                 max_len: int =256, 
                 d_ff: int =768, 
                 dropout: float=0.1, 
                 pad_id: int=0):
        super().__init__()
        self.pad_id = pad_id

        # Embeddings + positional embeddings
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = LearnablePositionalEmbedding(max_len=max_len, d_model=d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff,
            dropout=dropout, 
            batch_first=True, 
            norm_first=True  # norm_first=True = Pre-LN
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)

        # decoder_out -> vocab
        self.output_proj = nn.Linear(d_model, vocab_size)

    
    def forward(self, 
                memory: torch.Tensor,
                report_ids: torch.Tensor, # initial or generated texts
                mem_mask: torch.Tensor = None):
        
        B, T = report_ids.shape
        device = report_ids.device

        # Embedd repprt tplems
        x = self.emb(report_ids)  # report ids like <start> or <impression> etc..
        x = self.pos(x)

        # Masks
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=report_ids.device).bool()
        pad_mask = (report_ids == self.pad_id)  # (B, T) — PyTorch expects this shape here
        
        # Decode
        out = self.decoder(
            tgt=x,          # Query
            memory=memory,  # Key, Value
            tgt_mask=causal_mask,           # (T, T) causal
            tgt_key_padding_mask=pad_mask,  # (B, T) padding
            memory_key_padding_mask = mem_mask # Will always be None

        )
        logits = self.output_proj(self.ln_f(out))
        return logits
    