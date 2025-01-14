import torch.nn as nn
from transformers import AutoModel


class Verifier(nn.Module):
    def __init__(self, model_name, dropout=0.4):
        super(Verifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, 1) # binary output indicating whether if a text is written by the user or not

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask).last_hidden_state[:,0,:]
        x = self.dropout(x)
        x = self.linear(x)

        return x

