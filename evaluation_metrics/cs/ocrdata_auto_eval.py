# -*- coding: utf-8 -*-


import csv
import json
from statistics import mean, stdev
import sys
import re




titlePath='title_cs.txt'
dataPath='data_cs.txt'
goldPath = 'gold_cs.txt'
generatedPath = 'vlt5s_cs.txt'


fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']

count = 0

generatedScores = []
#baselineScores = []
untemplatedScores = [1,1]




with open(generatedPath, encoding='utf-8') as generatedFile:
    gen_file = generatedFile.readlines()
    
    

    

with open(dataPath, 'r', encoding='utf-8') as dataFile, open(titlePath, 'r', encoding='utf-8') as titleFile, \
        open(goldPath, 'r', encoding='utf-8') as goldFile:
    for datas, titles, gold in zip(dataFile.readlines(), titleFile.readlines(), goldFile.readlines()):
        dataArr = datas.split()
        titleArr = titles.split()
        goldArr = gold.split()
        recordList = []
        for gld in goldArr:
            data_string = datas.replace("_", " ")
            if gld.lower() in " ".join([data_string,titles]).lower()  and gld.lower() not in fillers and gld.lower() not in recordList:
                recordList.append(gld.lower())
        list1 = recordList
        list2 = recordList
        list3 = recordList
        recordLength = len(recordList)
        generatedList = []
        summary1 = gen_file[count]
        
            
        for token in summary1.split():
            if token.lower() in list1:
                list1.remove(token.lower())
                generatedList.append(token.lower())

       
        count += 1
        
        if recordLength==0:
            generatedRatio=0
        else:
            generatedRatio = len(generatedList) / recordLength
        

        generatedScores.append(generatedRatio)
        

        

print(f'generated CS stdev: {round(stdev(generatedScores)*100,2)}%')

print()
print(f'generated CS mean: {round(mean(generatedScores)*100,2)}%')

print()
print(f'generated CS RSD: {round((stdev(generatedScores)*100) / abs(mean(generatedScores)),2)}%')

    
    
    
    
    