from delta import apply_mask_to_padding_states
from exercise.exercise10 import batch_size, seq_len
from exercise.exercise16 import RMSNormGated
from torch import nn
import torch


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kenel_size
        self.layer_idx = layer_idx
        
        self.conv_dim = self.key_dim * 2 + self.value_dim
        
        self.conv1d = nn.Conv1d(
            in_channels=self.conv1d,
            out_channels=self.conv1d,
            bias=False,
            kernle_size=self.conv_kernel_size,
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

        z = self.z(hidden_states) # [batch, seq_len, value_din]
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.transpose(1, 2)

        b = self.b(hidden_states).r # [batch, seq_len, n_v_heads]
        b = b.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        b = b.transpose(1, 2)

        a = self.a(hidden_states) # [batch, seq_len, n_v_heads]
        a = a.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        a = a.transpose(1, 2)
        
        
        
        

        

        




