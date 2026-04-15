from click.core import V
import torch
import torch.nn.functional as F
import torch.nn as nn

from delta import torch_recurrent_gated_delta_rule
from utils import apply_interleaved_mrope

"""
self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
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
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.num_v_heads
        self.num_k_heads = config.num_k_heads
        self.head_v_dim = config.head_v_dim
        self.head_k_dim = config.head_k_dim
        self.k_dim = self.num_k_heads * self.head_k_dim
        self.v_dim = self.num_v_heads * self.head_v_dim
        self.kernel_size = config.linear_conv_kernel_size

        self.norm = RMSNorm(self.head_v_dim, config.rms_norm_eps)

        self.conv_dim = self.k_dim * 2 + self.v_dim
        self.qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.kernel_size - 1
        )
        
        self.z = nn.Linear(self.hidden_size, self.v_dim, bias=False)
        self.b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
    
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=False)
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache = None
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        if attention_mask is not None:
            if attention_mask.shape[1] != hidden_states.shape[1]:
                attention_mask = attention_mask[:, -hidden_states.shape[1] :]
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape

        recurrent_state = cache.recurrent_state[self.layer_idx] if cache is not None else None
        conv_state = cache.conv_state[self.layer_idx] if cache is not None else None


        mixed_qkv = self.qkv(hidden_states) # [batch, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, conv_dim, seq_len]

        if cache is not None:
            pad_len = max(self.kernel_size - mixed_qkv.shape[-1], 0)
            conv_state = F.pad(mixed_qkv, (pad_len, 0))
            cache.conv_state[self.layer_idx] = conv_state
        
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]) # [batch, conv_dim, seq_len]
        mixed_qkv = mixed_qkv.transpose(1, 2) # [batch, seq_len, conv_dim]

        # conv_dim: 2 * k_dim + v_dim

        q, k, v = torch.split(
            mixed_qkv,
            [self.k_dim, self.k_dim, self.v_dim],
            dim=-1
        )

        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)


        z = self.z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.b(hidden_states) # [batch, seq_len, num_v_heads]
        a = self.a(hidden_states) # [batch, seq_len, num_v_heads]

        beta = torch.sigmoid(b)
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            


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
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

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
        super().__init__()
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

    

def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos: [batch, seq_len, pos_size]
    # sin: [batch, seq_len, pos_size]

    cos = cos.unsqueeze(1) # [batch, 1, seq_len, pos_size]
    sin = sin.sunsqueeze(1) # [batch, 1, seq_len, pos_size]


    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_emebd = (cos * q_pass) + (rotate_half(q_rot) * sin)
    k_emebd = (sin * k_pass) + (rotate_half(k_rot) * sin)

    q = torch.cat((q_emebd, q_pass), dim=-1)
    k = torch.cat((k_emebd, k_pass), dim=-1)

    return q, k


class SelfAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):  
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_head
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)

        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_embddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache = None
    ):
        # hidden_states: [batch, seq_len, hidden_size]
        # pos_emd: [batch, seq_len, pos_size]

        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states) # [batch, seq, self.num_attention_heads * self.head_dim * 2]
        q, gate = torch.chunk(q_proj, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2) # [batch, seq_len, attn_head, head_dim]
        k = self.k(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # [batch, seq_len, kv_heads, head_dim]
        v = self.v(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # [batch, seq_len, kv_heads, head_dim]

        cos, sin = pos_embddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)


        



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
    
    hidden_states = torch.randn((batch_size, seq_len, config.hidden_size))
    attention_mask = torch.randn((batch_size, seq_len))

    delta_net = GatedDeltaNet(config, 1)
    out = delta_net(hidden_states, attention_mask)
    print(out.shape)

    pos_ids = torch.arange(0, seq_len)[None, :].expand(batch_size, -1).float()
    rope = RoPE(config)
    out = rope(hidden_states, pos_ids)
    print(out)