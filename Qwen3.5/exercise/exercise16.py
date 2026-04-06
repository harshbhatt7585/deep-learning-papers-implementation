# implement entire model

from re import S
from tokenize import group
from attention import Qwen35Attention
import attention
from delta import apply_mask_to_padding_states
from exercise.exercise10 import batch_size, hidden_states, position_embeddings, seq_len
from exercise.exercise5 import torch_casual_conv1d_update
from mlp import Qwen35MLP
from norm import Qwen35RMSNorm
from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        out = self._norm(x)
        out = out * (1.0 + self.weight)
        return out.to(dtype=x.dtype)

    

class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float= 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps 
    
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int 
    ):
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_group = self.num_k_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** 0.5


        self.query = nn.Linear(
            self.hidden_size, 
            self.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias
        )

        self.key = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        self.value = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )


    
        self.norm = RMSNorm(self.hidden_size, config.rms_norm_eps)

        self.out_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        positon_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_param = None
    ):
        # hidden_states: [batch, seq_len, hidden_size]
        # position_ids: [batch, seq_len]
        # attention_mask: [batch, seq_len]
        batch_size, seq_len, hidden_size = hidden_states.shape

        query_proj = self.query(hidden_states) #  [batch, seq_len, self.num_attention_heads * self.head_dim * 2]
        query_states, gate = torch.chunk(query_proj, dim=-1)
        
        query_states = query_states.reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim) # [batch, seq_len, num_attention_heads, head_dim]
        gate = gate.reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)  # [batch, seq_len, num_attention_heads, head_dim]

        query_states = query_states.transpose(1, 2) # [batch, attention_heads, seq_len, head_dim]

        key_states = self.key(hidden_size) # [batch, seq_len, num_kv_heads * head_dim]
        key_states = key_states.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        key_states = key_states.transpose(1, 2) # [batch,  kv_heads, seq_len, head_dim]

        value_states = self.value(hidden_size)
        value_states = value_states.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.transpose(1, 2) # [batch, kv_heads, seq_len, head_dim]

        if cache_param is not None:
            key_states, value_states = cache_param.update(key_states, value_states)
        
        attn_weight = query_states @ key_states.transpose(1, 2)
        attn_weight = attn_weight * self.scaling
        attn_weight = torch.softmax(attn_weight, dim=-1)







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
    


class DynamicCache:
    def __init__(
        self,
        config
    ):
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "fulL_attention"
        ]
        
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_sates = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for  _ in range(config.num_hidden_layers)]

    
    def update(
        self,
        key_states,
        value_states,
        layer_idx
    ):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        else:
            self.key_cache[layer_idx] = torch.cat(self.key_cache[layer_idx], key_states)
            self.value_cache[layer_idx] = torch.cat(self.value_cache[layer_idx], value_states)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    
    @property
    def has_previous_state(self) -> bool:
        return self.conv_states[self.last_linear_layer] is not None

    
    def get_seq_len(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        return self.key_cache[layer_idx].shape[-2]


    
        

class Decoder(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        
        self.layer_type = config.layer_types[layer_idx]
        
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config=config, layer_idx=layer_idx)
        
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config, layer_idx)
        
        else:
            raise ValueError("Not suported!")
        
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        postion_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value=None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_value,
                attention_mask=attention_mask
            )
        
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=postion_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states



def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size, 
    query_length,
    kv_length,
    device,
    dtype
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min

    causal = torch.full((query_length. kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + kv_length - query_length) # causal: [query_lrngth, kv_length]
    
    
    
    causal = causal[None, None, ...].expand(batch_size, 1, query_length, kv_length)
    if attention_mask is None:
        return causal
    
    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value
    return causal + padding_mask




class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values = None,
        use_cache: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        device: str = 'cpu'
    ) -> tuple[torch.Tensor, DynamicCache | None]:
        
        batch_size, seq_len, _ = input_ids.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        position_ids = torch.arange(seq_len, device=device) + past_seen_tokens        
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_key_values),
                device=device,
                dtype=inputs_embeds.dtype
            )
        
        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=device,
            dtype=inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
    
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


if __name__ == "__main__":
    from types import SimpleNamespace
    config = SimpleNamespace(
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
    )

    model = Qwen35TextModel(config)

    input_ids = torch.randint((1, 32))
    input_embds = torch.randn((1, 128))

    


    
