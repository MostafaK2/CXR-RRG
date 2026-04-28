from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class ClinicalTextEncoder(nn.Module):
    def __init__(
        self,
        model_name:    str   = "emilyalsentzer/Bio_ClinicalBERT",
        d_model:       int   = 512,
        dropout:       float = 0.1,
        freeze_layers: int   = 0,
        max_length:    int   = 128,
    ):
        
        super().__init__()

        self.max_length = max_length
        self.d_model    = d_model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        bert_d = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(bert_d, d_model),
            nn.LayerNorm(d_model), 
            nn.Dropout(dropout)
        )
        
    
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n_layers: int):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, clinical_texts, device):

        # Get the clinical text tokens (padded and truncated)
        encoded = self.tokenizer(
            clinical_texts,
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        ).to(device) 

        # Bert Forward 
        out = self.bert(**encoded) 
        token_embeds = out.last_hidden_state
        # project, layernorm and dropout
        token_embeds = self.projection(token_embeds)
        # attn_mask
        attn_mask = encoded["attention_mask"] == 0
        return token_embeds, attn_mask


# if __name__ == "__main__":
#     import torch
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     encoder = ClinicalTextEncoder(
#         model_name    = "emilyalsentzer/Bio_ClinicalBERT",
#         d_model       = 512,
#         dropout       = 0.1,
#         freeze_layers = 12,
#         max_length    = 128,
#     ).to(DEVICE)

#     # Parameter summary
#     trainable     = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
#     non_trainable = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
#     dummy_texts = [
#         "Patient presents with shortness of breath and chest pain.",
#         "No acute cardiopulmonary process identified."
#     ]
    
#     with torch.no_grad():
#         token_embeds, attn_mask = encoder(dummy_texts, DEVICE)

#         # print(f"  output shape:          {token_embeds.shape}")        # [2, 128, 512]
#         # print(f"  attn_mask shape:       {attn_mask.shape}")           # [2, 128]
#         # print(f"  output norm (total):   {token_embeds.norm():.4f}")
#         # print(f"  output norm per token: {token_embeds.norm() / (token_embeds.shape[0] * token_embeds.shape[1]) ** 0.5:.4f}")
#         # print(f"  output min:            {token_embeds.min():.4f}")
#         # print(f"  output max:            {token_embeds.max():.4f}")
#         # print(f"  masked tokens:         {attn_mask.sum().item()} padding positions")

