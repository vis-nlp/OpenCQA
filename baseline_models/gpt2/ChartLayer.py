# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import torch



class ChartLayerModule(nn.Module):
    def __init__(self, model):
        super(ChartLayerModule, self).__init__()
        
        self.base_model = model
        self.chart_embed = nn.Linear(50257, 50257) #linearly project krl chart input to match embed size
        
    
    def forward(self, inputs_embeds):
       model_output = self.base_model(inputs_embeds = inputs_embeds)[0]
       model_output2 = self.chart_embed(model_output)
       return model_output2




