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


def rotate_half(x: torch.Tensor):
    # x: [batch, heads, seq_len, dim]

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2: ]
    return torch.cat((-x1, x2), dim=-1)



def apply_rope(
    self, 
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_paas = k[..., :rot_dim], k[..., rot_dim:]

    # q_rot: [batch, heads, seq_len, rot_dim]

    q_emebd = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emebd = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_emebd = torch.cat((q_emebd, q_pass), dim=-1)
    k_emebd = torch.cat((k_emebd, k_paas), dim=-1)

    return q_emebd, k_emebd

