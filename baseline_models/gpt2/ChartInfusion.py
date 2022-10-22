# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import torch

#module extracts the embedding layer from gptneo,adds the charts embeddings to it,and replaces
#the embedding layer with this module

class ChartInfusingModule(nn.Module):
    def __init__(self, wte, embed_size = 2560, chart_dim = 100):
        super(ChartInfusingModule, self).__init__()
        # self.pretrained = my_pretrained_model
        self.chart_dim = chart_dim
        self.chart_embed = nn.Linear(chart_dim, embed_size) #linearly project krl chart input to match embed size
        self.embed = wte #replace embedding layer from gptneo with this one
    
    def forward(self, chart_infused_model_input):
        chart_embeddings = chart_infused_model_input[:,0:self.chart_dim,:];print(chart_embeddings)
        model_input = chart_infused_model_input[:,self.chart_dim:,0];print(model_input);
        model_input=model_input.long()
        chart_embeddings = self.chart_embed(chart_embeddings)
        model_input = self.embed(model_input)
        print(model_input.shape);
        print(chart_embeddings.shape);
        assert model_input.shape == chart_embeddings.shape
        return torch.cat([chart_embeddings,model_input], 1)
        




# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
# new_module = ChartInfusingModule(model.transformer.wte,chart_dim=2)
# model.transformer.wte = new_module
# print(model.transformer.wte)

# a= torch.rand([2, 2, 2560])
# b= torch.randint(10, (2, 2))
# b_zeros = torch.zeros(2, 2)
# b_pad = torch.stack([b,b_zeros])
# c=torch.cat([a,b_pad], 1)
# d= model(inputs_embeds = a)





# b_embed = model.transformer.wte(b)
# c=torch.cat([a,b_embed], 1)
# d= model(inputs_embeds = c)


# model.transformer.wte.chart_embed(torch.rand([2, 2]))
# new_module.chart_embed(a).shape


