# -*- coding: utf-8 -*-

import json

with open("pew/bertqa_data/targetAnswers.txt", 'r',encoding='utf8') as actualfile:
            actual = actualfile.readlines()
            

with open("pew/bertqa_data/generatedAnswers.txt", 'r',encoding='utf8') as generatedfile:
            generated = generatedfile.readlines()
    
output_list_actual = []
output_list_generated = []
count = 0;
for i,j in zip(actual,generated):
    data_actual = { "image_id":count, "caption":i }; 
    data_generated = { "image_id":count, "caption":j };
    count+=1;
    output_list_actual.append(data_actual)
    output_list_generated.append(data_generated)
    #json_data = json.dumps(data)
    # output_dict['id'] = count; count+=1;
    # output_dict['text'] = i;
    # output_dict['claim'] = j;


      
with open('pew/bertqa_data/targetAnswers.json','w') as f:
    json.dump(output_list_actual,f)


with open('pew/bertqa_data/generatedAnswers.json','w') as f:
    json.dump(output_list_generated,f)




# with open(input_file, "r", encoding="utf-8") as f:
#            a = json.load(f)