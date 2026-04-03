import torch
from types import SimpleNamespace


class Qwen35DynamicCache:
    def __init__(self, config):
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    
    def update(self, key_states, value_states, layer_idx):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(self.key_cache[layer_idx].device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(self.value_cache[layer_idx].device))
            if self.conv_states[layer_idx] is not None:
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(self.conv_states[layer_idx].device))
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx.to(self.conv_states[layer_idx].device))
        
    

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.conv_states[layer_idx] is not None

    def update_conv_state(self, conv_state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        self.conv_states[layer_idx] = conv_state
        return conv_state

    def update_recurrent_state(self, recurrent_state: torch.Tensor | None, layer_idx: int) -> torch.Tensor | None:
        self.recurrent_states[layer_idx] = recurrent_state
        return recurrent_state


    @property
    def has_any_previous_state(self) -> bool:
        return self.conv_states[self.last_linear_layer] is not None


if __name__ == "__main__":
    a = torch.randn(2, 4, 5, 32)
    config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
              "linear_attention",
              "full_attention",
              "linear_attention",
              "full_attention",
          ],
        
    )
    cache = Qwen35DynamicCache(config)
    print(a.shape)
