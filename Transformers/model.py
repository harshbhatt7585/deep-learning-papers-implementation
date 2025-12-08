import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)

        position = torch.arrange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        ).view(1, -1, 1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
        self.scale = d_model ** 0.5
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = torch.matmul(q, k.transpose(-1, -2))
        attn = torch.softmax(scores / self.scale, dim=-1)
        output = torch.matmul(attn, v)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerBlock, self).__init__()
        self.attn = SelfAttention(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_vocab, max_len):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_vocab = n_vocab
        self.max_len = max_len

        self.embedding = nn.Embedding(n_vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.attention = SelfAttention(d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        
        return x



    