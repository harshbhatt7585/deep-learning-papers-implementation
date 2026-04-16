"""
self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

beta = torch.sigmoid(b)
g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)
"""

import torch
from torch import nn
import torch.nn.functional as F

from delta import torch_recurrent_gated_delta_rule
from exercise.exercise16 import MLP
from mask import build_causal_mask
from utils import apply_interleaved_mrope


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
    
    def forward(
        self,
        x: torch.Tensor
    ):
        out = self._norm(x.float())
        out = out * (1.0 + self.weight.float()) 
        return out.to(dtype=x.dtype)



class RoPE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

        self.dim = min(config.dim, config.head_dim)
        self.pos_rotary_factor = 1.0
        self.theta = config.theta * self.pos_rotary_factor

        # [dim // 2]
        inv_freq = 1.0 / (
            self.theta ** torch.arange(0, self.dim, 2) / self.dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    
    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor
    ):
        # x: [batch, num_heads, seq_len, head_dim] 
        # pos_ids: [batch, seq_len]
    

        # [3, batch, seq_len]
        pos_ids = pos_ids[None, ...].expand(
            3,
            pos_ids.shape[0],
            -1
        )

        # [3, batch, dim // 2, 1]
        inv_freq = self.inv_freq[None, None, :, None].expand(
            3,
            pos_ids.shape[1],
            -1,
            1
        )

        # [3. batch, dim // 2, 1] x [3, batch, 1, seq_len] = [3, batch, dim // 2, seq_len]
        freq = inv_freq @ pos_ids[:, :, None, :].float() 
        freq = freq.transpose(2, 3) # [3, batch, seq_len, dim // 2]

        freq = apply_interleaved_mrope(freq) # [batch, seq_len, dim // 2]
        embd = torch.cat((freq, freq), dim=-1) # [batch, seq_len, dim]

        # [batch, seq_len, dim]
        return embd.cos().to(dtype=x.dtype), embd.sin().to(dtype=x.dtype) 


def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), dim=-1)
    

def apply_rotary_pos_emd(
    q,
    k,
    cos,
    sin
):
    # q: [batch_size, num_attn_heads, seq_len, head_dim]
    # cos: [batch_size, seq_len, pos_dim]

    cos = cos.unsqueeze(1) # [batch_size, 1, seq_len, pos_dim]
    sin = sin.unsqueeze(1) # [batch_size, 1, seq_len, pos_dim]

    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_emebd = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emebd = (k_rot * cos) + (rotate_half(k_rot) * sin)


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

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        
        
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)


        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias
        )
        
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)

        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.scaling = self.head_dim ** -0.5

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        cache = None
    ):
        # hidden_states: [batch, seq_len, hidden_size]
        # pos_embeddings: [batch, seq_len, pos_dim]
        # attention_mask/causal: [batch, num_heads, query_length, kv_length]

        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states) 
        q, gate = torch.chunk(q_proj, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        # [batch_size, num_attn_heads, seq_len, head_dim]
        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = pos_embeddings
        q, k = apply_rotary_pos_emd(q, k, cos, sin)


        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        
        # compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v) # [batch, num_heads, seq_len dim]
        attn_out = attn_out.transpose(1, 2) # [batch, seq_len, num_heads, dim]

        attn_out = attn_out.reshape(batch_size, seq_len, -1) # [batch, seq, hidden_size]
        attn_out = attn_out * torch.sigmoid(gate)

        out = self.o_proj(attn_out)
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
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim

        self.kernel_size = config.linear_conv_kernel_size

        self.k_dim = self.num_k_heads * self.head_k_dim
        self.v_dim = self.num_v_heads * self.head_v_dim


        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))


        self.conv_dim = self.k_dim * 2 + self.v_dim
        
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim,
            bias=False
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.conv_dim,
            bias=False
        )

        self.out_proj = nn.Linear(
            self.v_dim,
            self.hidden_size,
            bias=False
        )

        self.in_proj_z = nn.Linear(
            self.hidden_size,
            self.v_dim,
            bias=False
        )

        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False
        )
        
        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False
        )
    

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        cache = None
    ):
        
        # hidden_states: [batch, seq_len, hidden_states]
        # attention_mask: [batch, seq_len]
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None: 
            if attention_mask.shape[1] != hidden_states.shape[1]:
                attention_mask = attention_mask[:, -hidden_states.shape[1] :]
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        qkv = self.in_proj_qkv(hidden_states) # [batch, seq_len, conv_dim]
        qkv = qkv.transpose(1, 2)  # [batch, conv_dim, seq_len]

        conv_state = cache.conv_state[self.layer_idx] if cache is not None else None
        recurrent_state = cache.recurrent_state[self.layer_idx] if cache is not  None else None

        if cache is not None:
            conv_pad = max(self.kernel_size - qkv.shape[-1], 0)
            conv_state = F.pad(qkv, (conv_pad, 0))
            cache.conv_state[self.layer_idx] = conv_state

        qkv = F.silu(self.conv1d(qkv)[:, :, :seq_len]).transpose(1, 2) # [batch, seq_len, conv_dim]

        q, k, v = torch.split(qkv, [self.k_dim, self.k_dim, self.v_dim], dim=-1) 

        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        z = self.in_proj_z(hidden_states) # [batch, seq_len, v_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states) # [batch, seq_len, num_v_heads]
        a = self.in_proj_a(hidden_states) # [batch, seq_len, num_v_heads]

        beta = torch.sigmoid(b) 
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)


        attn_core, recurrent_state = torch_recurrent_gated_delta_rule(
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

        attn_core = attn_core.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        attn_core = self.norm(attn_core)
        attn_core = attn_core * F.silu(z)
        attn_core = attn_core.reshape(batch_size, seq_len, self.v_dim)

        out = self.out_proj(attn_core)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, self.layer_idx)
        
        else:
            self.self_attn = SelfAttention(config, self.layer_idx)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
        self.mlp = MLP(config)

    
    def forward(
        self,
        hidden_states,
        pos_embeddings,
        attn_mask = None,
        cache = None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states,
                attn_mask,
                cache
            )       

        else:
            hidden_states = self.self_attn(
                hidden_states,
                pos_embeddings,
                attn_mask,
                cache
            )
        
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states




class DynamicCache(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.num_layers = config.num_hidden_layers
        self.recurrent_state = [None for _ in range(self.num_layers)]
        self.conv_state = [None for _ in range(self.num_layers)]
        self.k = [None for _ in range(self.num_layers)]
        self.v = [None for _ in range(self.num_layers)]

        self.transformer_layers = [i for i in range(config.num_hidden_layers) if config.layer_types[i] == "full_attention"]

    
    def update(self, k, v, layer_idx: int):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        
        else:
            self.k[layer_idx] = torch.cat((self.k[layer_idx], k), dim=-2)
            self.v[layer_idx] = torch.cat((self.v[layer_idx], v), dim=-2)
        
        return self.k[layer_idx], self.v[layer_idx]


    def get_seq_len(self):
        return self.k[self.transformer_layers[0]].shape[-2] if self.k[self.transformer_layers[0]] is not None else 0

    def get_seq_length(self):
        return self.get_seq_len()




class TextModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.config = config


        self.rope = RoPE(config)
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Identity()
    
    def forward(
        self,
        input_ids=None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = True,
    ):
        # input_ids: [batch, seq_len]
        # inputs_embeds: [batch, seq_len, hidden_size]

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).to(device=input_ids.device)

        batch_size, seq_len, _ = inputs_embeds.shape

        cache = past_key_values
        if use_cache and cache is None:
            cache = DynamicCache(self.config)

        past_seen_tokens = cache.get_seq_len() if cache is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        kv_length = seq_len + past_seen_tokens
        
        causal_mask = build_causal_mask(
            attention_mask,
            batch_size,
            seq_len,
            kv_length,
            inputs_embeds.device,
            inputs_embeds.dtype
        )

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids[None, ...].expand(batch_size, -1)
        
        positional_embedding = self.rope(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                positional_embedding,
                mask,
                cache
            )
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states, cache
        




if __name__ == "__main__":
    from config import config

    norm = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    # print(out)

    batch_size = 4
    seq_len = 20

    pos_ids = torch.arange(0, seq_len).expand(batch_size, -1)
    rope = RoPE(config)
    out = rope(torch.randn(batch_size, seq_len, 128), pos_ids)
    # print(out)

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos_emd = torch.randn(batch_size, seq_len, config.head_dim)

    attn = SelfAttention(config, 1)
    out = attn(
        hidden_states,
        (pos_emd, pos_emd)
    )
    print(out.shape)

    delta_net = GatedDeltaNet(config, 1)
    out = delta_net(
        hidden_states,
    )
    print(out.shape)

    decoder = Decoder(config, 1)
    out = decoder(
        hidden_states,
        (pos_emd, pos_emd)
    )

    model = TextModel(config)
    out = model(
        torch.ones((batch_size, seq_len), dtype=torch.long),
    )
    
