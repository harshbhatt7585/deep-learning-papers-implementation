# implement rotary positional embedding

import torch
from torch import nn

class RotaryPosEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_dim_factor = config.pos_dim_factor
        self.dim = config.dim

        self.inv_freq = 1.0 / (config.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
    
    def forward(self, x: torch.Tensor, positon_ids: torch.Tensor):
        
        if positon_ids.dim() == 2:
            positon_ids = positon_ids[None, ...].expand(3, positon_ids.shape[0], -1)
        
        inv_feq_expanded = self.inv_freq[None, None, :, None].float().expand(
            positon_ids.shape[0],
            positon_ids.shape[1],
            -1,
            1
        )

        freq = inv_feq_expanded @ positon_ids[:, :, None, :].float()
        emb = torch.cat((freq, freq), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


if __name__ == "__main__":
    from types import SimpleNamespace
    config = SimpleNamespace(
        pos_dim_factor=1,
        theta=1,
        dim=20
    )
    model = RotaryPosEmbedding(config)
    print(model(
        torch.randn(4, 4, 20, 20),
        torch.arange(0, 20).unsqueeze(0).expand(4, -1)
    ))

        
