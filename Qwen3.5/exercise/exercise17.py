from delta import apply_mask_to_padding_states
from exercise.exercise16 import RMSNormGated
from torch import nn
import torch


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_size
        self.layer_idx = layer_idx
        
        self.conv_dim = self.key_dim * 2 + self.value_dim
        
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )

        
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = RMSNormGated(self.head_k_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim,
            bias=False
        ) 

        self.z = nn.Linear(
            self.hidden_size,
            self.value_dim,
            bias=False
        )

        
        self.b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False
        )

        self.a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False
        )

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_param = None,
        attention_mask = None
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states=hidden_states, attention_mask=attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        
        use_precomputed_cache = cache_param is not None and cache_param.has_previous_state() and seq_len == 1
        conv_state = cache_param.conv_state[self.layer_idx] if cache_param is not None else None
        recurrent_state = cache_param.recurrent_state[self.layer_idx] if cache_param is not None else None


        mixed_qkv = self.qkv(hidden_states) # [batch, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, conv_dim, seq_din]

        if use_precomputed_cache:
            mixed_qkv = self.conv1d(mixed_qkv)
        
        else:
            pass
            # implement it for training
        
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.z(hidden_states) # [batch, seq_len, value_din]
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.transpose(1, 2)

        b = self.b(hidden_states) # [batch, seq_len, n_v_heads]
        a = self.a(hidden_states) # [batch, seq_len, n_v_heads]


        query, key, value = torch.chunk(mixed_qkv, 3, dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, -1)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, -1)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, -1)

        beta = torch.sigmoid(b)
        g = -torch.exp(self.A_log) * torch.softplus(a + self.dt_bias)


        if use_precomputed_cache:
            attn_core_out, recurrent_state = delta_rule(
                recurrent_state,
                query,
                key,
                value,
                beta,
                g,
                cache_param
            ) 

        
        attn_core_out = self.norm(attn_core_out, z)
        attn_core_out = attn_core_out.transpose(1, 2).contigious()
        attn_core_out = attn_core_out.reshape(batch_size, seq_len, self.hidden_size)

        out = self.out_proj(attn_core_out)
        return out


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        hidden_size=128,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        intermediate_size=256,
        num_v_heads=4,
        num_k_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        linear_conv_kernel_size=4,
        rotaty_factor=1.0,
        theta=10000.0,
        dim=32,
        mrope_section=[11, 11, 10],
    )

    model = GatedDeltaNet(
        config=config,
        layer_idx=1
    )

    model(
        torch.randn((4, 20, 128))
    )
        

        

        




