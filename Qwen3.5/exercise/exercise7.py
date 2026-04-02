
from torch import nn


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        self.hidden_size = config.hidden_size
        self.head_v_dim = config.head_v_dim
        self.head_k_dim = config.head_k_dim
        self.num_v_heads = config.num_v_heads
        self.num_k_heads = config.num_k_heads

        self.key_dim = config.key_dim
        self.value_dim = config.value_dim


        self.mixed_qkv = nn.Linear(self.hidden_size, self.hidden_size * 2 + self.key_dim, bias=True)
        self.z = nn.Linear(self.hidden_size, self.key_dim, bias=True)
        self.b = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        self.a = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)

        
        self.dt = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(self.num_v_heads))
        self.norm = Qwen35RMSNorm(self.head_v_dim, eps=config.eps)
        self.out_proj = nn.Linear(self.head_v_dim, self.hidden_size)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_dim - 1
        )

    def forward(self, x: torch.Tensor, cache_params=None, attention_mask=None):
        hidden_states =  apply_mask_to_padding_states(hidden_states, attention_mask)
        
        # batch, seq, dim
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_cache = cache_params is not None and cache_params.has_previous_state and seq_len == 1
        conv_state = cache_params.conv_state[self.layer_idx] if cache_params is not None else None
        recurrent_state = cache_params.recurrent_state[self.layer_idx] if cache_params is not None else None

        mixed_qkv = self.mixed_qkv(hidden_states) #[batch, seq_len, hidden_size * 2 + key_dim]
        mixed_qkv = mixed_qkv.trasnpose(1, 2) # [batch, hidden_size * 2 + key, seq_len]
        z = self.z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.b(hidden_states)
        a = self.b(hidden_states)

        if use_precomputed_cache:
            mixed_qkv = torch_causal_conv1d_update(
                hidden_states=mixed_qkv,
                conv_state=hidden_states,
                weight=self.conv.weight,
                bias=self.conv.bias
            )

        else:
            if cache_params is not None:
               conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
               cache_params.conv_state[self.layer_idx] = conv_state

            
            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_v_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)


            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            
            if self.num_v_heads // self.num_k_heads > 1:
                rep = self.num_v_heads // self.num_k_heads
                query = query.repeat_interleave(rep, dim=2)
                key = key.repeat_interleave(rep, dim=2)

            
            if use_precomputed_cache:
                core_attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                    cache_params is not None else None
                )
            
            else:
                core_attn_out, recurrent_state = torch_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                    cache_params is not None else None
                )
                

               



    
