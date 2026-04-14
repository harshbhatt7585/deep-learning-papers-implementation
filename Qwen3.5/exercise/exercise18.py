import torch
import torch.nn.functional as F
import torch.nn as nn


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
        self.head_dim = config.head_dim
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

    

        
    

        




class RoPE(nn.Module):
    def __init__(
        self,
        config
    ):
        pass



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



