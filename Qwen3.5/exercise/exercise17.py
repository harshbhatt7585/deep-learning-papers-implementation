from math import e
from turtle import pos
from delta import (
    apply_mask_to_padding_states,
    torch_causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)

from norm import Qwen35RMSNorm
from torch import nn
import torch.nn.functional as F
import torch


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: int = 1e-6
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        out = self._norm(x)
        out = out * (1.0 + self.weight.float())
        return out.to(dtype=x.dtype)






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
        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
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
        attn_core_out = self.norm(attn_core_out)
        attn_core_out = attn_core_out * F.silu(z)
        attn_core_out = attn_core_out.reshape(batch_size, seq_len, -1)

        out = self.out_proj(attn_core_out)
        return out

    
class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
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
        q = q.transpose(1, 2) # [batch, num_attn_head, seq, head_dim]
        gate = gate.reshape(batch_size, seq_len, -1) # [batch, seq, num_attn_heads * 2]

        k = self.k(hidden_states)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v(hidden_states)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, v = apply_rotary_pos_emb(q, v, cos, sin)


        if past_key_value:
            k, v = past_key_value.update(q, v)
        
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)


        # attention computation
        attn_weight = torch.matmul(k, v.transpose(2, 3))
        attn_weight = attn_weight * self.scaling
        
        if attention_mask is not None:
            attn_weight = attn_weight + attention_mask

        attn_weight = torch.softmax(attn_weight, dim=-1)

        attn_out = torch.matmul(attn_weight, v)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(batch_size, seq_len, hidden_size)
        attn_out = attn_out * torch.sigmoid(gate)

        out = self.out_proj(attn_out)
        return out


class RopE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.rotraty_factor = 1.0
        self.theta = config.theta
        self.dim = config.head_dim

        
        # [dim]
        self.inv_freq = 1.0 / (
            self.theta ** torch.arange(0, self.dim, 2) / self.dim
        )

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positon_ids: torch.Tensor
    ):
        # hidden_satets: [batch, seq, dim]
        # positon_ids: [batch, seq_len]
        if positon_ids.ndim == 2:
            positon_ids = positon_ids[None, ...].expand(3, positon_ids.shape[0], -1)
        
        
        # we want to expand this dimension of inv_freq to work with positon_ids
        # [3, batch_size, din // 2, 1]
        expanded_inv_freq = self.inv_freq[None, None, ..., None].float().expand(
            positon_ids.shape[0],
            positon_ids.shape[1],
            -1,
            1
        )

        # [3, batch, dim // 2, seq_len]
        freq = expanded_inv_freq @ positon_ids[:, :, None, :]
        # [3, batch, seq_len, dim // 2]
        freq = freq.transpose(2, 3)

        # [3, batch, seq_len, dim]
        embd = torch.cat((freq, freq), dim=-1)
        return embd.cos().to(dtype=hidden_states.dtype), embd.sin().to(dtype=hidden_states.dtype)



def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    # cos: [batch_size, seq_len, pos_dim]
    # sin: [batch_size, seq_len, pos_dim]
    # q: [batch, atten_head, seq_len, head_dim]
    # k: [batch, kv_head, seq_len, head_dim]

    cos = cos.unsqueeze(1) # [batch, 1, seq_len, pos_dim]
    sin = sin.unsqueeze(1) # [batch, 1, seq_len, pos_dim]

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed



def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int, 
    kv_length: int,
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, dtype=dtype, device=device) # [query_length, kv_length]
    causal = torch.triu(causal, diagonal= 1 + kv_length - query_length)  # [query_length, kv_length]

    causal = causal[None, ...].expand(batch_size, 1, query_length, kv_length) # [batch, 1, q_len, kv_len]
    if attention_mask is None:
        return causal

    # attention_mask: [batch_size, seq_len]
    
    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value
    return padding_mask + causal



class MLP(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x




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
            self.linear_attention = GatedDeltaNet(config=config, layer_idx=layer_idx)

        elif self.layer_type == "self_attention":
            self.self_attention = Attention(config=config, layer_idx=layer_idx)
        
        else:
            pass
        
        self.mlp = MLP(config=config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        postional_embedding: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value = None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attnetion":
            hidden_states = self.linear_attention(
                hidden_states,
                past_key_value,
                attention_mask
            )
        elif self.layer_type == "self_attention":
            hidden_states = self.self_attention(
                hidden_states,
                postional_embedding,
                attention_mask,
                past_key_value
            )
        else:
            pass
            
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class DynamicCache(nn.Module):
    def __init__(
        self,
        config
    ):
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_state = [None for _ in range(config.num_hidden_layers)]

    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int
    ):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=-1)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value), dim=-1)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]





class TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = RopE(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value = None,
        use_cache: bool = False,
        input_embeds: torch.Tensor | None = None,
    ):
        if input_embeds is None:
            input_embeds = self.embedding(input_ids)

        if use_cache and past_key_value is None:
            past_key_value = DynamicCache(config)
        
        batch_size, seq_len, hidden_size = input_embeds.shape

        past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0

        if position_ids is None:
            pos_ids = torch.arange(0, seq_len, dtype=input_embeds.dtype, device=input_embeds.device) + past_seen_tokens
            pos_ids = pos_ids[None, ...].expand(batch_size, -1)

        pos_emb = self.rope(input_embeds, pos_ids)
        

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                device=input_embeds.device,
                dtype=input_embeds.dtype
            )
            
        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask,
            batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=input_embeds.device,
            dtype=input_embeds.dtype
        )

        hidden_states = input_embeds

        for layer in self.layers:
            layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states=hidden_states,
                postional_embedding=pos_emb,
                attention_mask=layer_mask,
                past_key_value=past_key_value
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_value

        

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

    cos = torch.ones(batch_size, 1, config.head_dim)
    sin = torch.zeros(batch_size, 1, config.head_dim)

    attn = Attention(config, 1)
    out = attn(hidden_states, (cos, sin))
    print(out.shape)
        
    
    rope = RopE(config)
    pos_ids = torch.randn(batch_size, 20)
    out = rope(hidden_states, pos_ids)
    # print(out)

    decoder = Decoder(config, 0)
    out = decoder(hidden_states, (cos, sin))
    print(out.shape)

    llm = TextModel(config)
    input_ids = torch.ones((batch_size, 20), dtype=torch.long)
    out = llm(
        input_ids=input_ids,
    )
    print(out)

