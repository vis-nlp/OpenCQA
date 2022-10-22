# -*- coding: utf-8 -*-



import pandas as pd
import re
import json
import random




def load_json_file(filePath):
    with open(filePath,encoding='utf8') as f:
        return json.load(f)


def save_json_file(targetPath,output):
    with open(targetPath, 'w',encoding='utf8') as f:
        json.dump(output, f)
  



def load_text_file(filePath):
    with open(filePath,'r',encoding='utf8') as f:
        return f.read()



def find_article(img_path,img_no):
    
    
    if "no_data" in img_path:
        article = load_text_file("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/no_data/article_scrape/" + img_no + ".txt")
    elif "two_col" in img_path:
        article = load_text_file("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/two_col/article_scrape/" + img_no + ".txt")
    elif "multi_col" in img_path:
        article = load_text_file("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/multi_col/article_scrape/" + img_no + ".txt")

    else:
        raise Exception("file not found ",img_path)
        
        # row = out_multicol_metadata[ out_multicol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        # if len(row) == 0:
        #     row = out_twocol_metadata[ out_twocol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        # bbox_path = row['bboxesPath'].item()
    
    
    return article



def random_selection(summary):
    summ_len = len(summary.split())
    overlap = 0.41
    end_token = int(summ_len - summ_len * overlap)+1
    random_start_token = random.randint(0,end_token)
    return " ".join(summary.split()[random_start_token:random_start_token+summ_len - end_token])


all_annots = pd.read_csv("../all_annotations.csv",encoding='utf8',index_col=0)



train = load_json_file("../data/train.json")
test = load_json_file("../data/test.json")
val = load_json_file("../data/val.json")
# full = load_json_file("../data/full_dataset.json")




for dataset,name in [(train,'train'),(test,'test'),(val,'val')]:
    for key in dataset:
        # if "\n" in dataset[key][2]:
        #     raise Exception("found error quote")
        # img_path = all_annots.loc[int(key.replace(".png",""))]['img_path']
        # article = find_article(img_path,re.findall(r'\d+', img_path)[0])
        # article = article.replace("\n"," ")
        # dataset[key] = dataset[key][:2] + [article] + dataset[key][2:]
        dataset[key][2] = random_selection(dataset[key][2])
    save_json_file(name + "_randsumm.json",dataset)


train.update(val)
train.update(test) 
save_json_file('full_dataset_randsumm.json',train)

randsumm_list = []

for key in train:
    randsumm_list.append(train[key][2])

all_annots['random_summary'] = randsumm_list

all_annots.to_csv("all_annotations_randsumm.csv",encoding='utf8', line_terminator='\n')



# ratio=0
# for index,row in all_annots.iterrows():
#         ratio+=len(row['random_summary'].split())/len(row['summary'].split())
#         #ratio+=len(row['extractive_answer'].split())/len(row['article'].split())
# size = all_annots.shape[0]
# print(f'average percentage of tokens extracted : {round((ratio/size)*100)}%')













