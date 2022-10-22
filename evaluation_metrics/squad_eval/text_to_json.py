# -*- coding: utf-8 -*-

import json




# gold_path = "data/test.json"
# predict_path = "output/test2/predictions.json"

output_path = "pew/gpt2_data/pred.json" #generated summaries
predicted_path = "pew/gpt2_data/generatedAnswers.txt"
gold_path = "pew/gpt2_data/test.json"

output = {}




gold = json.load(open(gold_path, 'r'))

with open(predicted_path,'r',encoding='utf8') as f:
    predicted = f.readlines()


# processed_data = []
counter=0
for id in gold:
 output[id] = predicted[counter];counter+=1;

  
  
with open(output_path, 'w') as outfile:
    json.dump(output, outfile)
  
  
  
  