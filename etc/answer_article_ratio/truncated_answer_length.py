# -*- coding: utf-8 -*-

import pandas as pd


data = pd.read_csv("T5_Q_T_OCR_article.csv",encoding='utf8')
retention_rates=[]
for index,row in data.iterrows():
    # article = row['Context'][row['Context'].find('Context:'):]
    input_ = row['Context']
    input_ = input_.split()
    truncated_input = input_[:512] #first 512 tokens
    truncated_input = set(truncated_input)
    answer = row['Summary']
    answer = answer.split()
    answer_tokens = []
    for token in answer:
        if token in truncated_input:
            answer_tokens.append(token)
    retention_rates.append(len(answer_tokens)/len(answer))

print(sum(retention_rates)/len(retention_rates))
    