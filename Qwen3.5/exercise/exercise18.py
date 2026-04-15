from click.core import V
import torch
import torch.nn.functional as F
import torch.nn as nn

from delta import torch_recurrent_gated_delta_rule
from utils import apply_interleaved_mrope

"""
self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads)._uniform(0, 16)))

beta = torch.sigmoid(b)
g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)
"""


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: int = 1e-6 
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def _norm(self, x: torch.Tensor):
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        out = self._norm(x.float())
        out = out * (1 + self.weight.float())
        return out


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.num_v_heads
        self.num_k_heads = config.num_k_heads
        self.head_v_dim = config.head_v_dim
        self.head_k_dim = config.head_k_dim
        self.k_dim = self.num_v_heads * self.head_dim
        self.v_dim = self.num_v_heads * self.head_dim
        self.kernel_size = config.linear_conv_kernel_size

        self.norm = RMSNorm(self.hidden_size, config.rms_norm_eps)

        self.conv_dim = self.k_dim * 2 + self.v_dim
        self.qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.kernel_size,
            groups=self.conv1d
        )
        
        self.z = nn.Linear(self.hidden_size, self.v_dim, bias=False)
        self.b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads)._uniform(0, 16)))
    
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=False)
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache = None
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        if attention_mask.shape[1] != hidden_states.shape[1]:
            attention_mask = attention_mask[:, -hidden_states[1] :]
        
        hidden_states = hidden_states * attention_mask[:, :, None]

        batch_size, seq_len, _ = hidden_states.shape

        recurrent_state = cache.recurrent_state[self.layer_idx] if cache is not None else None
        conv_state = cache.conv_state[self.layer_idx] if cache is not None else None


        mixed_qkv = self.qkv(hidden_states) # [batch, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, conv_dim, seq_len]

        if cache is not None:
            pad_len = max(self.kernel_size - mixed_qkv.shape[-1], 0)
            conv_state = F.pad(mixed_qkv, (pad_len, 0))
            cache.conv_state[self.layer_idx] = conv_state
        
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, seq_len]) # [batch, conv_dim, seq_len]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, seq_len, conv_dim]

        # conv_dim: 2 * k_dim + v_dim

        q, k, v = torch.split(
            mixed_qkv,
            [self.k_dim, self.k_dim, self.v_dim]
        )

        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.hidden_size).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.hidden_size).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.hidden_size).transpose(1, 2)


        z = self.z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.b(hidden_states) # [batch, seq_len, num_v_heads]
        a = self.a(hidden_states) # [batch, seq_len, num_v_heads]

        beta = torch.sigmoid(b)
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)

        core_attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            recurrent_state,
            cache is not None
        )

        if cache is not None:
            cache.recurrent_state[self.layer_idx] = recurrent_state

        # flatten
        core_attn_out = core_attn_out.reshape(-1, self.num_v_heads)
        z = z.reshape(-1, self.num_v_heads)

        core_attn_out = self.norm(core_attn_out)
        core_attn_out = core_attn_out * F.silu(z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        
        out = self.out_proj(core_attn_out)
        return out




class RoPE(nn.Module):
    def __init__(
        self,
        config
    ):
        self.dim = config.head_dim
        self.theta = config.theta

        # [dim // 2]
        inv_freq = 1.0 / (
            self.theta ** torch.arange(0, self.dim, 2) / self.dim
        )

        self.register_buffer('inv_freq', inv_freq)


    
    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor
    ):

        # x: [batch, seq, hidden_size]
        # pos_ids: [batch. seq_len]

        # [3, batch, seq_len]
        pos_ids = pos_ids[None, ...].expand(3, pos_ids.shape[0], -1)

        # [3, batch, dim // 2, 1]
        inv_freq = self.inv_freq[None, None, ..., None].expand(
            3,
            pos_ids.shape[1],
            -1,
            1
        ).to(device=x.device)

        # [3, batch, dim // 2, seq_len]
        freq = inv_freq @ pos_ids[:, :, None, :] 

        # [3, batch, seq_len, dim // 2]
        freq = freq.transpose(2, 3)

        # [batch, seq_len, dim // 2]
        freq = apply_interleaved_mrope(freq)

        # [batch, seq_len, dim]
        emebd = torch.cat((freq, freq), dim=-1)
        
        cos = emebd.cos().to(dtype=x.dtype, device=x.device)
        sin = emebd.cos().to(dtype=x.dtype, device=x.device)

        return cos, sin
        

    

        



class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        pass


class Decoder(nn.Module):
    def __init__(
        self, 
        config,
        layer_idx: int
    ):
        pass



class DynamicCache(nn.Module):
    def __init__(
        self,
        config
    ):
        pass


class TextModel(nn.Module):
    def __init__(
        self,
        config
    ):
        pass

        

if __name__ == "__main__":
    from config import config
    rms = RMSNorm(config.hidden_size)

    batch_size = 4
    seq_len = 20
    
    out = rms(torch.randn(batch_size, seq_len, config.hidden_size))
    print(out.shape)



