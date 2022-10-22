# -*- coding: utf-8 -*-



import json
import math

ocr_dir = "../../../bboxes/"
generated_path ="generatedAnswers_bertqa.txt"
reference_path = "../data/test.json"

with open(generated_path,'r',encoding='utf8') as f:
    generated = f.readlines()

with open(reference_path,'r',encoding='utf8') as f:
    test = json.load(f)



def calc_recall_precision_cef(line_no,tokens,ocr):
    ocr_text = " ".join([key['sentence'] for key in ocr])
    ocr_text = " ".join(list(filter(lambda x: x not in fillers, set(ocr_text.split()))))
    matches=0
    if len(tokens)==0:
        return 0
    for token in tokens:
        if token.lower() in ocr_text.lower():
            matches+=1
    return matches/len(ocr_text.split()) , matches/len(tokens)


def calc_recall_cef(line_no,tokens,ocr):
    region_scores = []  #extraction score for each region
    weighted_region_scores = []
    lens_ = []
    for index,key in enumerate(ocr):
        counter = 0 # number of matches in each region
        for token in tokens:
            if token.lower() in key['sentence'].lower():
                counter +=1
        len_ = len(key['sentence'].split())
        if len_==0:
            print("Current ocr region empty: ", line_no)
            continue
        region_scores.append((counter/len_))
        weighted_region_scores.append((counter/len_) * math.exp(len_))
        lens_.append(len_)
    sum_ = sum(weighted_region_scores)
    if sum_==0:
      print("Current summary extracts no ocr elements: ", line_no)
      return 0
    else:
        max_ = max(weighted_region_scores)
        normalized_region_scores = [i/max_ for i in weighted_region_scores]
        
    return sum(normalized_region_scores)/len(normalized_region_scores)







def calc_coverage_cef(line_no,tokens,ocr):
    lookup=[0]*len(ocr)
    for token in tokens:
        for index,key in enumerate(ocr):
            if token.lower() in key['sentence'].lower():
                lookup[index]=1
    return sum(lookup)/len(ocr)


fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']

#micro_ief=0
coverage_cef=0
recall_cef=0
precision_cef=0

for index,key in enumerate(test):
    file_no = test[key][0]
    with open(ocr_dir+file_no.replace(".png",".json"),'r',encoding='utf8') as f:
        ocr = json.load(f)
    summary = generated[index]
    tokens = list(filter(lambda x: x not in fillers, set(summary.split())))
    #micro_ief+=calc_micro_ief(index,tokens,ocr)
    coverage_cef+=calc_coverage_cef(index,tokens,ocr)
    tuple_ = calc_recall_precision_cef(index,tokens,ocr)
    recall_cef += tuple_[0]
    precision_cef += tuple_[1]

print(generated_path)
print("Chart Extraction Factor(Recall): ", recall_cef/(index+1))
print("Chart Extraction Factor(Coverage): ", coverage_cef/(index+1))
print("Chart Extraction Factor(Precision):: ", precision_cef/(index+1))







