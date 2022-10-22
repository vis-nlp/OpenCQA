# -*- coding: utf-8 -*-




# from sklearn import utils
import os
import pandas as pd
import shutil
import json
from sklearn import utils
import re
import copy as cp

dataset = {}
offset_path = "../../"
target_image_path = offset_path+"chart_images/"
source_image_path = offset_path+"../chart2text_extended/c2t_dataset_pew(not_complete)/out/"
full_dataset_target_path = offset_path+"data/full_dataset.json"
train_dataset_target_path = offset_path+"data/train.json"
test_dataset_target_path = offset_path+"data/test.json"
val_dataset_target_path = offset_path+"data/val.json"

target_bbox_path = offset_path+"bboxes/"
source_bbox_path = offset_path+"../chart2text_extended/c2t_dataset_pew/dataset/"
#metadata for no_data folder and root out folder
out_multicol_metadata = pd.read_csv(offset_path+"../chart2text_extended/c2t_dataset_pew/dataset/multiColumn/metadata.csv",encoding='utf8')
out_twocol_metadata = pd.read_csv(offset_path+"../chart2text_extended/c2t_dataset_pew/dataset/metadata.csv",encoding='utf8')


#metadata for no_data_remaining folder
remain_multicol_metadata = pd.read_csv(source_image_path + "no_data_remaining/multi_col/metadata.csv",encoding='utf8')
remain_twocol_metadata = pd.read_csv(source_image_path + "no_data_remaining/two_col/metadata.csv",encoding='utf8')





#keep adding to the list when new batches are resolved and ready
batch_list = ["../mturk_study/batches/disagreements_resolved/combined_final/batch_7.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_8.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_9.csv",
                "../mturk_study/batches/disagreements_resolved/combined_final/batch_10.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_11.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_12.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_13.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_14.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_15.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_16.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_17.csv",
                "../mturk_study/batches/disagreements_resolved/combined_final/batch_18.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_19.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_20.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_21.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_22.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_23.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_24.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_25.csv",
                "../mturk_study/batches/disagreements_resolved/combined_final/batch_26.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_27.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_28.csv",
               "../mturk_study/batches/disagreements_resolved/combined_final/batch_29.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_30.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_31.csv",
              "../mturk_study/batches/disagreements_resolved/combined_final/batch_32.csv"
              ]

combined_frame = pd.DataFrame()

for batch in batch_list:
    combined_frame = pd.concat([combined_frame,pd.read_csv(batch)])

combined_frame.reset_index(drop=True, inplace=True)
combined_frame_filtered= cp.copy(combined_frame)




filter_list = ['mturk_charts/two_col/44.png','mturk_charts/no_data/two_col/2883.png','mturk_charts/no_data/multi_col/8646.png',
'mturk_charts/no_data/multi_col/4801.png','mturk_charts/no_data/multi_col/6028.png','mturk_charts/no_data/multi_col/6195.png',
'mturk_charts/no_data/multi_col/6563.png','mturk_charts/no_data/multi_col/6635.png','mturk_charts/no_data/multi_col/6639.png',
'mturk_charts/no_data/multi_col/6998.png','mturk_charts/no_data_remaining/multi_col/1076.png','mturk_charts/no_data/multi_col/6028.png',
'mturk_charts/no_data/multi_col/24181.png','mturk_charts/no_data/multi_col/24270.png','mturk_charts/no_data_remaining/two_col/20.png',
'mturk_charts/no_data_remaining/two_col/28.png','mturk_charts/no_data_remaining/two_col/33.png','mturk_charts/no_data_remaining/two_col/131.png',
'mturk_charts/no_data_remaining/two_col/170.png','mturk_charts/no_data_remaining/two_col/194.png','mturk_charts/no_data_remaining/two_col/208.png',
'mturk_charts/no_data_remaining/two_col/212.png','mturk_charts/no_data_remaining/two_col/251.png','mturk_charts/no_data_remaining/two_col/278.png',
'mturk_charts/no_data_remaining/two_col/306.png','mturk_charts/no_data_remaining/two_col/467.png','mturk_charts/no_data_remaining/two_col/468.png',
'mturk_charts/no_data_remaining/two_col/535.png','mturk_charts/no_data_remaining/two_col/598.png','mturk_charts/no_data_remaining/two_col/620.png',
'mturk_charts/no_data_remaining/multi_col/160.png','mturk_charts/no_data/multi_col/7471.png','mturk_charts/no_data/multi_col/7676.png',
'mturk_charts/no_data/multi_col/7763.png','mturk_charts/no_data/multi_col/8240.png','mturk_charts/no_data/multi_col/8362.png',
'mturk_charts/no_data/multi_col/8646.png','mturk_charts/no_data_remaining/multi_col/1372.png','mturk_charts/no_data_remaining/multi_col/1392.png',
'mturk_charts/no_data_remaining/multi_col/1444.png','mturk_charts/no_data_remaining/multi_col/1459.png','mturk_charts/no_data_remaining/multi_col/1474.png',
'mturk_charts/no_data_remaining/multi_col/1539.png','mturk_charts/no_data_remaining/multi_col/1542.png','mturk_charts/no_data_remaining/multi_col/1594.png',
'mturk_charts/no_data_remaining/multi_col/1595.png','mturk_charts/no_data_remaining/multi_col/1634.png','mturk_charts/no_data_remaining/multi_col/1639.png',
'mturk_charts/no_data_remaining/multi_col/1680.png','mturk_charts/no_data_remaining/multi_col/1683.png','mturk_charts/no_data_remaining/multi_col/1875.png',
'mturk_charts/no_data_remaining/multi_col/2229.png','mturk_charts/no_data_remaining/multi_col/2308.png','mturk_charts/no_data_remaining/multi_col/2327.png',
'mturk_charts/no_data_remaining/multi_col/2438.png','mturk_charts/no_data_remaining/multi_col/2524.png','mturk_charts/no_data_remaining/multi_col/2539.png',
'mturk_charts/no_data_remaining/multi_col/2545.png','mturk_charts/no_data_remaining/multi_col/2556.png','mturk_charts/no_data_remaining/multi_col/2570.png',
'mturk_charts/no_data_remaining/multi_col/2577.png','mturk_charts/no_data_remaining/multi_col/2589.png','mturk_charts/no_data_remaining/multi_col/2603.png',
'mturk_charts/no_data_remaining/multi_col/2634.png','mturk_charts/no_data_remaining/multi_col/2648.png','mturk_charts/no_data_remaining/multi_col/2673.png',
'mturk_charts/no_data_remaining/multi_col/2689.png','mturk_charts/no_data_remaining/multi_col/2693.png','mturk_charts/no_data_remaining/multi_col/2696.png',
'mturk_charts/no_data_remaining/multi_col/2764.png','mturk_charts/no_data_remaining/multi_col/2791.png','mturk_charts/no_data_remaining/multi_col/2830.png',
'mturk_charts/no_data_remaining/multi_col/2837.png','mturk_charts/no_data_remaining/multi_col/2854.png']



for index,row in combined_frame.iterrows():
    if row['img_path'] in filter_list:
        combined_frame_filtered = combined_frame_filtered.drop(labels=index, axis=0)
    
combined_frame_filtered.reset_index(drop=True, inplace=True)



combined_frame_filtered.to_csv("all_annotations.csv",encoding='utf8')

counter=0


def find_bbox_path(source_image,img_no):
    
    if "out/no_data/multi_col" in source_image:
        row = out_multicol_metadata[ out_multicol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        if len(row) == 0:
            row = out_twocol_metadata[ out_twocol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        bbox_path = row['bboxesPath'].item()
    elif "out/no_data/two_col" in source_image:
        row = out_twocol_metadata[ out_twocol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        if len(row) == 0:
            row = out_multicol_metadata[ out_multicol_metadata['old_id'] == 'no_data-'+img_no[:-len('.png')]]
        bbox_path = row['bboxesPath'].item()
    elif "out/multi_col" in source_image:
        row = out_multicol_metadata[ out_multicol_metadata['old_id'] == 'multi_col-'+img_no[:-len('.png')]]
        if len(row) == 0:
            row = out_twocol_metadata[ out_twocol_metadata['old_id'] == 'two_col-'+img_no[:-len('.png')]]
        bbox_path = row['bboxesPath'].item()
    elif "out/two_col" in source_image:
        row = out_twocol_metadata[ out_twocol_metadata['old_id'] == 'two_col-'+img_no[:-len('.png')]]
        if len(row) == 0:
            row = out_multicol_metadata[ out_multicol_metadata['old_id'] == 'multi_col-'+img_no[:-len('.png')]]
        bbox_path = row['bboxesPath'].item()
    elif "out/no_data_remaining/multi_col" in source_image:
        old_id = remain_multicol_metadata[ remain_multicol_metadata['id'] == int(img_no[:-len('.png')])]['old_id'].item()
        row = out_multicol_metadata[ out_multicol_metadata['old_id'] == old_id]
        if len(row) == 0:
            row = out_twocol_metadata[ out_twocol_metadata['old_id'] == old_id]
        bbox_path = row['bboxesPath'].item()
    elif "out/no_data_remaining/two_col" in source_image:
        old_id = remain_twocol_metadata[ remain_twocol_metadata['id'] == int(img_no[:-len('.png')])]['old_id'].item()
        row = out_twocol_metadata[ out_twocol_metadata['old_id'] == old_id]
        if len(row) == 0:
            row = out_multicol_metadata[ out_multicol_metadata['old_id'] == old_id]
        bbox_path = row['bboxesPath'].item()
    
    return bbox_path







for index,row in combined_frame_filtered.iterrows():
    relative_path = row['img_path'][len('mturk_charts/'):row['img_path'].rfind('/')+1] + "imgs/"
    img_no = row['img_path'][row['img_path'].rfind('/')+1:]
    source_image = source_image_path + relative_path + img_no
    target_image = target_image_path + str(counter) + ".png"
    shutil.copy(source_image,target_image)
    
    bbox_path = find_bbox_path(source_image,img_no)
    source_bbox = source_bbox_path + bbox_path
    target_bbox = target_bbox_path + str(counter) + ".json"
    shutil.copy(source_bbox,target_bbox)
    
    
    dataset[counter] =  [
                         str(counter)+".png",
                         row['title'],
                         row['summary'],
                         row['question'],
                         row['abstractive_answer'],
                         row['extractive_answer']
                         ]
    
    counter+=1






with open(full_dataset_target_path, 'w', encoding='utf8') as f:
    json.dump(dataset, f)



# shuffle data with seed=0 for reproducibility
indicesShuffled = utils.shuffle(list(dataset.keys()) , random_state=0)

trainSize = round(len(indicesShuffled) * 0.7)
testSize = round(len(indicesShuffled) * 0.15)
validSize = len(indicesShuffled) - trainSize - testSize


train_data = {}
val_data = {}
test_data = {}


for file_no in range(0,trainSize):
    train_data[indicesShuffled[file_no]] = dataset[indicesShuffled[file_no]]


for file_no in range(trainSize,trainSize + validSize):
    val_data[indicesShuffled[file_no]] = dataset[indicesShuffled[file_no]]


for file_no in range(trainSize + validSize,len(indicesShuffled)):
    test_data[indicesShuffled[file_no]] = dataset[indicesShuffled[file_no]]


with open(train_dataset_target_path, 'w', encoding='utf8') as f:
    json.dump(train_data, f)

with open(val_dataset_target_path, 'w', encoding='utf8') as f:
    json.dump(val_data, f)

with open(test_dataset_target_path, 'w', encoding='utf8') as f:
    json.dump(test_data, f)





