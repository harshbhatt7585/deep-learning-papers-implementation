# implement rotary positional embedding

import torch

class RotaryPosEmbedding(nn.Module):
    def __init__(self, config):
        self.pos_dim_factor = config.pos_dim_factor
        self.dim = config.dim

        self.inv_freq = 1 / (config.theta ** torch.arange(0, self.dim, 2, dtype=torch.float()) / self.dim)
    
    def forward(self, x: torch.Tensor, positon_ids: torch.Tensor):
        
        if positon_ids.dim == 2:
            positon_ids = positon_ids[None, ...].expand(3, positon_ids.shape[0], -1)
        
        inv_feq_expanded = self.inv_freq[None, None, :, None].float().expand(
            positon_ids.shape[0],
            positon_ids.shape[1],
            -1,
            1
        )

        freq = inv_feq_expanded @ positon_ids
        emb = torch.cat((freq, freq), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

        

        
