import torch

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config,
        device,
    ):
        self.dim = config.dim
        self.factor = config.factor
        self.theta = config.theta
        self.inv_freq = 1.0 /  ( self.theta ** (torch.arange(0, config.dim, 2, dtype=torch.float, device=device) / 2) )
    
    def forward(
        self,
        x: torch.Tensor,
        postion_ids: torch.Tensor
    ):
        # x: [batch, heads, seq_len, dim]
        # position_ids: [batch, ids]
        # inv_freq: [freq]

        if postion_ids.dim == 2:
            postion_ids = postion_ids[None, :, :].expand(3, x.shape[0], -1)
        
        
        inv_freq = self.inv_freq[None, None, None,  -1].expand(postion_ids.shape[0], postion_ids[1], 1, -1)
        
        freq = inv_freq @ postion_ids[:, :, None, :].float()
        # create pairs
        emb = torch.cat((freq, freq), dim=-1)
        return emb.cos(dtype=x.dtype), emb.sin(dtype=x.dtype)
        