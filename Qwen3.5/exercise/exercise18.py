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
        pass


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



