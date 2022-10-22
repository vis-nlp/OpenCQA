# -*- coding: utf-8 -*-



import pandas as pd
import re
import json





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
    
    if "no_data_remaining/multi_col" in img_path:
       if 'no_data-' in metadata_multicol.loc[int(img_no)]['old_id']:
           txt_no = re.findall(r'\d+', metadata_multicol.loc[int(img_no)]['old_id'])[0]
           article = load_text_file("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/no_data/article_scrape/" + txt_no + ".txt")
       else:
           raise Exception("file not in no_data article_scrape folder")
    elif "no_data_remaining/two_col" in img_path:
       if 'no_data-' in metadata_twocol.loc[int(img_no)]['old_id']:
           txt_no = re.findall(r'\d+', metadata_twocol.loc[int(img_no)]['old_id'])[0]
           article = load_text_file("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/no_data/article_scrape/" + txt_no + ".txt")
       else:
           raise Exception("file not in no_data article_scrape folder")
    elif "no_data" in img_path:
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



all_annots = pd.read_csv("../all_annotations.csv",encoding='utf8')
metadata_multicol = pd.read_csv("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/no_data_remaining/multi_col/metadata.csv",encoding='utf8',index_col='id')
metadata_twocol = pd.read_csv("../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/no_data_remaining/two_col/metadata.csv",encoding='utf8',index_col='id')


train = load_json_file("../data/train.json")
test = load_json_file("../data/test.json")
val = load_json_file("../data/val.json")
full = load_json_file("../data/full_dataset.json")


    

for dataset,name in [(train,'train'),(test,'test'),(val,'val'),(full,'full_dataset')]:
    for key in dataset:
        # if "\n" in dataset[key][2]:
        #     raise Exception("found error quote")
        img_path = all_annots.loc[int(key.replace(".png",""))]['img_path']
        article = find_article(img_path,re.findall(r'\d+', img_path)[0])
        article = article.replace("\n"," ")
        dataset[key] = dataset[key][:2] + [article] + dataset[key][2:]
    save_json_file(name + "_extended.json",dataset)




article_list = []

for key in full:
    article_list.append(full[key][2])

all_annots['article'] = article_list

all_annots.to_csv("all_annotations_extended.csv",encoding='utf8', index=False)

