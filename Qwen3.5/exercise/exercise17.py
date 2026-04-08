import attention
from delta import (
    apply_mask_to_padding_states,
    torch_causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)
from exercise.exercise16 import RMSNormGated
from norm import Qwen35RMSNorm
from rope import apply_rotary_pos_emb
from torch import nn
import torch.nn.functional as F
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
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
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
        cache_param=None,
        attention_mask=None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states=hidden_states, attention_mask=attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        
        conv_state = cache_param.conv_states[self.layer_idx] if cache_param is not None else None
        recurrent_state = cache_param.recurrent_states[self.layer_idx] if cache_param is not None else None

        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if cache_param is not None:
            pad_len = max(self.conv_kernel_size - mixed_qkv.shape[-1], 0)
            conv_state = F.pad(mixed_qkv, (pad_len, 0))
            cache_param.conv_states[self.layer_idx] = conv_state

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.z(hidden_states)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.b(hidden_states)
        a = self.a(hidden_states)


        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = torch.sigmoid(b)
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        attn_core_out, recurrent_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            cache_param is not None,
        )

        if cache_param is not None:
            cache_param.recurrent_states[self.layer_idx] = recurrent_state

        attn_core_out = attn_core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        attn_core_out = self.norm(attn_core_out, z)
        attn_core_out = attn_core_out.reshape(batch_size, seq_len, -1)

        out = self.out_proj(attn_core_out)
        return out

    
class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.scaling = self.head_dim ** 0.5

        self.q = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias
        )

        self.k = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )

        self.v = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )

        self.norm = Qwen35RMSNorm(self.hidden_size, config.rms_norm_eps)


        self.out_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )



    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple | None = None
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
 
        q_proj = self.q(hidden_states) # [batch, seq, num_attn_heads * head_dim * 2]
        q, gate = torch.chunk(q_proj, 2, dim=-1)
        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = q.tranpose(1, 2) # [batch, num_attn_head, seq, head_dim]
        gate = gate.reshape(batch_size, seq_len, -1) # [batch, seq, num_attn_heads * 2]

        k = self.k(hidden_states)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v(hidden_states)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        k, v = apply_rotary_pos_emb(cos, sin)


        if past_key_value:
            k, v = past_key_value.update(q, v)


        # attention computation
        attn_weight = torch.matmul(k, v.transpose(2, 3))
        attn_weight = attn_weight * self.head_dim
        attn_weight = torch.softmax(attn_weight)

        attn_out = torch.matmul(attn_weight, v)
        attn_out = attn_out.transpose(1, 2).contigious()
        attn_out = attn_out.reshape(batch_size, seq_len, hidden_size)
        attn_out = attn_out * torch.sigmoid(gate)

        out = self.out_proj(attn_out)
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

    batch_size = 4
    hidden_states = torch.randn(batch_size, 1, config.hidden_size)
    out = model(hidden_states)
    print(out.shape)
        

        

        


