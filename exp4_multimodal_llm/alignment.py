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