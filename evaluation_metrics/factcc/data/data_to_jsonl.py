# -*- coding: utf-8 -*-

import json

with open("pew/t5_data/targetAnswers.txt", 'r', encoding='utf-8') as actualfile:
            actual = actualfile.readlines()
            

with open("pew/t5_data/generated_predictions_sorted.txt", 'r', encoding='utf-8') as generatedfile:
            generated = generatedfile.readlines()
    
output_list = []
count = 0;
for i,j in zip(actual,generated):
    data = { "id":str(count), "text":i, "claim":j , "label":"INCORRECT"}; count+=1;
    output_list.append(data)
    #json_data = json.dumps(data)
    # output_dict['id'] = count; count+=1;
    # output_dict['text'] = i;
    # output_dict['claim'] = j;


      
with open('pew/t5_data/data-dev2.jsonl','w', encoding='utf-8') as f:
    json.dump(output_list,f)



# with open(input_file, "r", encoding="utf-8") as f:
#            a = json.load(f)