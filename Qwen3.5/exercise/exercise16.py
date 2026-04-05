# implement entire model

from re import S
from tokenize import group
from delta import apply_mask_to_padding_states
from exercise.exercise10 import batch_size, seq_len
from exercise.exercise5 import torch_casual_conv1d_update
from torch import nn
import torch

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx
    ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.num_v_heads
        self.num_k_heads = config.num_k_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.value_dim
        
        self.conv_kernel_size = config.linear_conv_kernel_size
        self.layer_idx = layer_idx
        
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size = self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(self.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # hidden_states: [batch, seq_len, hidden_size]
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
        conv_state = cache_params[self.layer_idx] if cache_params is not None else None
        recurrent_state = cache_params[self.layer_idx] if cache_params is not None else None 

        # we have to compute qkv, z, v, a and then use this to compute delta rule and outproj

        mixed_qkv = self.in_proj_qkv(hidden_states) # [batch, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, conv_dim, seq_len]

        z = self.in_proj_z(hidden_states) # [batch, seq_len, value_dim]
        z = z.resahpe(batch_size, seq_len, -1, self.head_v_dim)
        
        b = self.in_proj_b(hidden_states) # [batch, seq_len, num_v_heads]
        a = self.in_proj_a(hidden_states) # [batch, seq_len, num_v_heads]]

        if use_precomputed_states:
            mixed_qkv = torch_casual_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias
            )
        
        else:
            pass
            # implemented for training                 

        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, seq_len, seq_len]
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.resahpe(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(key, dim=2)
        
        if use_precomputed_states:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g, 
                beta,
                recurrent_state,
                cache_params is not None
            )
        else:
            pass 
            # implemented for training

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.resahpe(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)
    


class RoPE(nn.Module):
    def __init__(
        self,
        config
    ):
        self.rotary_factor = config.rotaty_factor
        self.theta = config.theta
        self.dim = config.dim

        self.inv_feq = 1.0 / (
            torch.arange(0, self.dim, 2) / self.dim
        )

    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ):
        # x: [batch, seq_len, dim]
        # pos: [batch, seq_len]

        # we need to convert pos into 3 times
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[-1], -1)

        # pos: [3, batch, seq_len]

        # inv_freq: [dim]
        
        inv_frq_expaded = self.inv_feq[None, None, :, None].float().expand(
            position_ids.shape[0], position_ids[1], -1, 1 
        )
        # inv_frq_expended: [3, batch, dim, 1]
        position_ids_expanded = position_ids[:, :, None, :].float() # [3, batch, 1, seq_len]
        freqs = (inv_frq_expaded @ position_ids_expanded) # [3, batch, dim, seq_len]
        freqs = freqs.transpose(2, 3) # [3, batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
        
        
