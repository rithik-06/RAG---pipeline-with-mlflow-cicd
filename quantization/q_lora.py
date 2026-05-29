import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModelForCausalLM
class LoRA(nn.Module):
    def __init__(self, model_name, r=4):
        super(LoRA, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(self.model.config.hidden_size, r))
        self.lora_B = nn.Parameter(torch.randn(r, self.model.config.hidden_size))

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        lora_output = hidden_states @ self.lora_A @ self.lora_B
        return lora_output + hidden_states