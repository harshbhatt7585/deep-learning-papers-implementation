import torch

def build_rope_cos_sin(
    position_ids,
    head_dim,
    theta=10000.0,
    partial_rotary_factor=1.0
):
    # postion_ids: [batch, seq_len]
    rotary_dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (theta ** torch.arange(0, rotary_dim, 2) / rotary_dim)

    freqs = position_ids[..., None] * inv_freq[None, None, :]

    return freqs


if __name__ == "__main__":
    print(build_rope_cos_sin(
        position_ids=torch.randn((4, 20)),
        head_dim=32,
    ).shape)