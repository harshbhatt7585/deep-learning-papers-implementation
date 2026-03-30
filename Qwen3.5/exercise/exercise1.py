import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2 ]
    x2 = x[..., x.shape[-1] // 2: ]

    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
    q,
    k,
    cos,
    sin
):
    # q, k: [batch_size, heads, seq_len, head_dim]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot

if  __name__ == "__main__":
    q = torch.randn(4, 8, 20, 32)
    k = torch.randn(4, 8, 20, 32)
    cos = torch.randn(4, 1, 20, 32)
    sin = torch.randn(4, 1, 20, 32)

    a = apply_rope(
        q,
        k, 
        cos,
        sin
    )
    print(a)