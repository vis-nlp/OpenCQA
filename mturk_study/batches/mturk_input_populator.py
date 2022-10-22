# -*- coding: utf-8 -*-


import pandas as pd
import os
from configparser import ConfigParser


# Construction 1
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)

# Construction 2
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer







dataset_dir = "../../../../chart2text_extended/c2t_dataset_pew(not_complete)/out/"

folder_list = ["two_col/",
               "multi_col/",
               "no_data/two_col/",
               "no_data/multi_col/",
               "no_data_remaining/two_col/",
               "no_data_remaining/multi_col/"]




img_dir = "imgs/"
caption_dir = "captions/"
title_dir = "titles/"


config = ConfigParser()
config.read('config.ini')




# batch_size = 30   #size of each mturk batch


batch_no = int(config.get('DEFAULT', 'batch_no')) #mturk batch number
# new_charts = 3 #number of new charts per hit

count = int(config.get('DEFAULT', 'last_loaded_file')) #count of image files 





input_csv = "inputs/input_" + str(batch_no) + ".csv"

new_batch = "new_unprocessed_batch.csv"


for folder in folder_list:
    a = os.listdir(dataset_dir+folder+img_dir)
    b = os.listdir(dataset_dir+folder+caption_dir)
    c = os.listdir(dataset_dir+folder+title_dir)
    assert len(a) == len(b) == len(c)
    for ele in range(len(a)):
        assert a[ele][:-4] == b[ele][:-4] == c[ele][:-4]


print("all image and caption and title files consistent")


all_images = []
all_captions = []
all_titles = []

for folder in folder_list:
    img_list = os.listdir(dataset_dir+folder+img_dir)
    [all_images.append("mturk_charts/" + folder + i) for i in img_list]    #refer to image paths on google firebase
    file_list = os.listdir(dataset_dir + folder + caption_dir)
    [all_captions.append(dataset_dir + folder + caption_dir + i) for i in file_list]  #refer to captions in local system
    title_list = os.listdir(dataset_dir + folder + title_dir)
    [all_titles.append(dataset_dir + folder + title_dir + i) for i in title_list]  #refer to captions in local system





input_table = pd.read_csv(input_csv,encoding='utf8')

new_batch_table = pd.read_csv(new_batch,encoding='utf8')



new_batch_table = new_batch_table[new_batch_table['category']=='good']
new_batch_table = new_batch_table.reset_index(drop=True)








    
def load_caption(caption):
    # with open(captionPath, 'r', encoding='utf-8') as captionFile:
    #     caption = captionFile.read()
    
    tokens = tokenizer(caption)
    tokenized_caption  = " ".join([i.text for i in tokens])
    

    return tokenized_caption    


def load_title(img_path):
    
    index = all_images.index(img_path)
    
    with open(all_titles[index], 'r', encoding='utf-8') as titleFile:
        title = titleFile.read() 

    return title    





batch_iter=0

try:
    assert len(input_table)*3 == len(new_batch_table)
except:
    raise Exception("New batch has too many or too few samples. Should have " + str(len(input_table)*3))    



for index,row in input_table.iterrows():
    input_table.loc[index ,'img_path4'] = new_batch_table['img_path'][batch_iter]
    input_table.loc[index ,'title4'] = load_title(new_batch_table['img_path'][batch_iter])
    input_table.loc[index ,'summary4'] = load_caption(new_batch_table['summary'][batch_iter]);batch_iter+=1;
    input_table.loc[index ,'img_path5'] = new_batch_table['img_path'][batch_iter]
    input_table.loc[index ,'title5'] = load_title(new_batch_table['img_path'][batch_iter])
    input_table.loc[index ,'summary5'] = load_caption(new_batch_table['summary'][batch_iter]);batch_iter+=1;
    input_table.loc[index ,'img_path6'] = new_batch_table['img_path'][batch_iter]
    input_table.loc[index ,'title6'] = load_title(new_batch_table['img_path'][batch_iter])
    input_table.loc[index ,'summary6'] = load_caption(new_batch_table['summary'][batch_iter]);batch_iter+=1;


count+=batch_iter;


input_table.to_csv(input_csv,encoding='utf8', index=False)

config.set('DEFAULT', 'last_loaded_file', str(count))



with open('config.ini', 'w') as f:
    config.write(f)









