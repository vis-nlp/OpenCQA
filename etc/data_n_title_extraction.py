# -*- coding: utf-8 -*-

import json


test_file = "../../data/test.json"


target_title_file = "testTitles.txt"
target_data_file = "testData.txt"

titles=[]
data=[]

with open(test_file, 'r') as f:
                test = json.load(f)

for key in test:
    titles.append(test[key][1])
    with open("../../bboxes/"+test[key][0].replace(".png",".json"), 'r') as f:
                bbox = json.load(f)
                
    data.append(" | ".join([row['sentence'].replace(" ","_") for row in bbox]))



with open(target_title_file, 'w',encoding='utf8') as f:
    for item in titles:
        f.write("%s\n" % item)


with open(target_data_file, 'w',encoding='utf8') as f:
    for item in data:
        f.write("%s\n" % item)