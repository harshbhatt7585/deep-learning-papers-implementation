import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config,
        device,
    ):
        super().__init__()
        self.dim = config.dim
        self.factor = config.factor
        self.theta = config.theta
        self.inv_freq = 1.0 /  ( self.theta ** (torch.arange(0, config.dim, 2, dtype=torch.float, device=device) / self.dim) )
    
    def forward(
        self,
        x: torch.Tensor,
        postion_ids: torch.Tensor
    ):
        # x: [batch, heads, seq_len, dim]
        # position_ids: [batch, ids]
        # inv_freq: [freq]

        if postion_ids.dim() == 2:
            postion_ids = postion_ids[None, :, :].expand(3, x.shape[0], -1)
        
        
        inv_freq = self.inv_freq[None, None, :, None].float().expand(postion_ids.shape[0], postion_ids.shape[1], -1, 1)
        
        freq = (inv_freq @ postion_ids[:, :, None, :].float()).transpose(2, 3)
        # create pairs
        emb = torch.cat((freq, freq), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor):
    # x: [batch, heads, seq_len, dim]

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), dim=-1)



def apply_rope(
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



if __name__ == "__main__":
    from types import SimpleNamespace
    config = SimpleNamespace(
        dim=20,
        factor=1.0,
        theta=10000
    )
    rope = RotaryEmbedding(config, 'cpu')
    
    q = torch.randn(4, 8, 20, 20)
    k = torch.randn(4, 8, 20, 20)
    position_ids = torch.arange(20).unsqueeze(0).expand(4, -1)
    cos, sin = rope(q, position_ids)

    print(apply_rope(q, k, cos, sin))

