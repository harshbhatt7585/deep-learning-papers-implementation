from __future__ import annotations


from torch import nn
from utils import ACT2FN

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


if __name__ == "__main__":
    from types import SimpleNamespace
    import torch
    config = SimpleNamespace(
        hidden_size=512,
        intermediate_size=100,
        hidden_act="silu"
    )
    model = Qwen3MLP(config)
    a = torch.randn((4, 512))
    print(model(a).shape)
    


