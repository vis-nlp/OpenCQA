# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import Counter
# from nltk.corpus import stopwords as nltkstopwords
# import nltk
# nltk.download('stopwords')

import gensim
gensim_stopwords = list(gensim.parsing.preprocessing.STOPWORDS)



#number of too complex,good,cannot create open ended question
######
unprocessed_dir = "../mturk_study/batches/unprocessed_batches/"
complex_count=0
good_count=0
unquestionable_count=0

for file in os.listdir(unprocessed_dir):
    batch = pd.read_csv(unprocessed_dir+file)
    complex_count+=batch[batch.category == 'complex'].shape[0]
    good_count+=batch[batch.category == 'good'].shape[0]
    unquestionable_count+=batch[batch.category == 'unquestionable'].shape[0]

print('good',good_count-76) #76 samples manually removed in analysis stage
print('too complex',complex_count)
print('cannot create open ended question',unquestionable_count+76) #76 samples manually removed in analysis stage
print('total',good_count+complex_count+unquestionable_count)
#######





#total,average number of tokens in summary,abstractive,extractive,question,title
#######
all_annotations = pd.read_csv("data(random_article_baseline)/all_annotations_randarticle.csv",encoding='utf8')
size = all_annotations.shape[0]
# col_list=['summary','random_summary','abstractive_answer','extractive_answer','question','title']
col_list=['summary','article','abstractive_answer','extractive_answer','question','title']


for col in col_list:
    total=0
    for index,row in all_annotations.iterrows():
        total+=len(row[col].split())
    print("total " + col + " tokens : " + str(total))
    print("average " + col + " tokens : " + str(total/size))

#average of ratio of number of tokens extracted to number of tokens in summary

ratio=0
for index,row in all_annotations.iterrows():
        ratio+=len(row['random_article'].split())/len(row['article'].split())
        #ratio+=len(row['extractive_answer'].split())/len(row['article'].split())

print(f'average percentage of tokens extracted : {round((ratio/size)*100)}%')


#percentage of answer1,answer2,new_answer selection to total number of answers

answer1_count=0
answer2_count=0
answer_new_count=0
answer_same_count=0
for index,row in all_annotations.iterrows():
    if row['answer_resolution'] == 'answer1':
        answer1_count+=1
    elif row['answer_resolution'] == 'answer2':
        answer2_count+=1
    elif row['answer_resolution'] == 'new_answer':
        answer_new_count+=1
    elif row['answer_resolution'] == 'same':
        answer_same_count+=1

print(f'percentage of answer1 selected : {round((answer1_count/size)*100)}%')
print(f'percentage of answer2 selected : {round((answer2_count/size)*100)}%')
print(f'percentage of new_answer selected : {round((answer_new_count/size)*100)}%')
print(f'percentage of same_answer selected : {round((answer_same_count/size)*100)}%')



#commong decontextualization tokens added and removed
text_added=[]
text_removed=[]

filter_list=['.','-',',', ')', '(']
filter_list = filter_list + gensim_stopwords

for index,row in all_annotations.iterrows():
    if str(row['decontextualizated_text_added']) != 'nan':
        for word in row['decontextualizated_text_added'].lower().split():
            if word not in filter_list:
                text_added.append(word)
    if str(row['decontextualizated_text_removed']) != 'nan':
        for word in row['decontextualizated_text_removed'].lower().split():
            if word not in filter_list:
                text_removed.append(word)
   

print(f'common texts added in decontextualization : {[word[0] for word in Counter(text_added).most_common(3)]}')
print(f'common texts removed in decontextualization : {[word[0] for word in Counter(text_removed).most_common(3)]}')




#######
        
        













