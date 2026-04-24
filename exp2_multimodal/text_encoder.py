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

        self.projection = nn.Linear(bert_d, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    
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
        token_embeds = self.drop(self.norm(token_embeds))

        # attn_mask
        attn_mask = encoded["attention_mask"] == 0

        return token_embeds, attn_mask


