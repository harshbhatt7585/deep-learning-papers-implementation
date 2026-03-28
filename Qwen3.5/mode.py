from torch import nn

class Qwen35CasualModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen35CasualModel(config)
        self.lm_model = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
    
    def forward(self, input_ids=None, attention_mask=None, positon_ids=None, past_key_values=None, inputs_emebds=None, use_cache=True):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positon_ids=positon_ids,
            past_key_values=past_key_values,
            inputs_emebds=inputs_emebds,
            use_cache=use_cache
        )
        logits = self.lm_model(hidden_states)
        return logits, past_key_values



