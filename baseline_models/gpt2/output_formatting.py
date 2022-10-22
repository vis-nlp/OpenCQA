# -*- coding: utf-8 -*-

import json




gold_path = "data/test.json"
predict_path = "outputs/full_dataset/qtc(tb)a/gpt2_0.00.json"

target_gold_path = "outputs/full_dataset/qtc(tb)a/targetAnswers.txt"
target_predict_path = "outputs/full_dataset/qtc(tb)a/generatedAnswers.txt"

target_gold = []
target_predict = []

with open(gold_path, 'r') as f:
                gold = json.load(f)

with open(predict_path, 'r') as f:
                predict = json.load(f)


for key in gold:
    target_gold.append(gold[key][5])
    target_predict.append(predict[key][0])



with open(target_gold_path, 'w',encoding='utf8') as f:
    for item in target_gold:
        f.write("%s\n" % item)


with open(target_predict_path, 'w',encoding='utf8') as f:
    for item in target_predict:
        f.write("%s\n" % item)