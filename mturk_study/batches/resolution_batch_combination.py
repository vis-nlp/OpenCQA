# -*- coding: utf-8 -*-


import math
import pandas as pd
import re


def find_extra_info(img,answer_resolution,extractive_answer):
        if len(df1.loc[df1['Input.img_path4'] == img]) != 0:
            title = df1.loc[df1['Input.img_path4'] == img]['Input.title4'].values[0]
            if answer_resolution=='same' or answer_resolution=='answer1':
                original_answer = df1.loc[df1['Input.img_path4'] == img]['Answer.answer4'].values[0]
            elif answer_resolution=='answer2':
                original_answer = df2.loc[df2['Input.img_path1'] == img]['Answer.answer1'].values[0]
            elif answer_resolution=='new_answer':
                original_answer = extractive_answer
            
        elif len(df1.loc[df1['Input.img_path5'] == img]) != 0:
            title = df1.loc[df1['Input.img_path5'] == img]['Input.title5'].values[0]
            if answer_resolution=='same' or answer_resolution=='answer1':
                original_answer = df1.loc[df1['Input.img_path5'] == img]['Answer.answer5'].values[0]
            elif answer_resolution=='answer2':
                original_answer = df2.loc[df2['Input.img_path2'] == img]['Answer.answer2'].values[0]
            elif answer_resolution=='new_answer':
                original_answer = extractive_answer
        
        elif len(df1.loc[df1['Input.img_path6'] == img]) != 0:
            title = df1.loc[df1['Input.img_path6'] == img]['Input.title6'].values[0]
            if answer_resolution=='same' or answer_resolution=='answer1':
                original_answer = df1.loc[df1['Input.img_path6'] == img]['Answer.answer6'].values[0]
            elif answer_resolution=='answer2':
                original_answer = df2.loc[df2['Input.img_path3'] == img]['Answer.answer3'].values[0]
            elif answer_resolution=='new_answer':
                original_answer = extractive_answer
        else:
            raise Exception("Could not find image",img)
        
        return title, original_answer


def find_question(img,answer_resolution,extractive_answer,resolved_question):
    
        if img in filter_list:
            return 0
        if len(df1.loc[df1['Input.img_path4'] == img]) != 0:
            title = df1.loc[df1['Input.img_path4'] == img]['Input.title4'].values[0]
            old_question = df1.loc[df1['Input.img_path4'] == img]['Answer.question4'].values[0]
            # if answer_resolution=='same' or answer_resolution=='answer1':
            #     original_answer = df1.loc[df1['Input.img_path4'] == img]['Answer.answer4'].values[0]
            #     new_question = df1.loc[df1['Input.img_path4'] == img]['Answer.question4'].values[0]
            # elif answer_resolution=='answer2':
            #     original_answer = df2.loc[df2['Input.img_path1'] == img]['Answer.answer1'].values[0]
            #     new_question = df2.loc[df2['Input.img_path1'] == img]['Answer.question1'].values[0]
            # elif answer_resolution=='new_answer':
            #     original_answer = extractive_answer
            
        elif len(df1.loc[df1['Input.img_path5'] == img]) != 0:
            title = df1.loc[df1['Input.img_path5'] == img]['Input.title5'].values[0]
            old_question =df1.loc[df1['Input.img_path5'] == img]['Answer.question5'].values[0]
            # if answer_resolution=='same' or answer_resolution=='answer1':
            #     original_answer = df1.loc[df1['Input.img_path5'] == img]['Answer.answer5'].values[0]
            # elif answer_resolution=='answer2':
            #     original_answer = df2.loc[df2['Input.img_path2'] == img]['Answer.answer2'].values[0]
            # elif answer_resolution=='new_answer':
            #     original_answer = extractive_answer
        
        elif len(df1.loc[df1['Input.img_path6'] == img]) != 0:
            title = df1.loc[df1['Input.img_path6'] == img]['Input.title6'].values[0]
            old_question =df1.loc[df1['Input.img_path6'] == img]['Answer.question6'].values[0]
            # if answer_resolution=='same' or answer_resolution=='answer1':
            #     original_answer = df1.loc[df1['Input.img_path6'] == img]['Answer.answer6'].values[0]
            # elif answer_resolution=='answer2':
            #     original_answer = df2.loc[df2['Input.img_path3'] == img]['Answer.answer3'].values[0]
            # elif answer_resolution=='new_answer':
            #     original_answer = extractive_answer
        else:
            raise Exception("Could not find image",img)
        
        
        if old_question == resolved_question:
            return 0
        else:
            try:
                if type(old_question)!=str and math.isnan(old_question):
                    old_question=str(old_question)
                tokens_common = set(old_question.split()).intersection(set(resolved_question.split()))
                tokens_removed = set(old_question.split())-tokens_common
            except Exception:
                raise Exception("error at ",old_question,resolved_question)
            global minor_change,stopword_removal
            for tok in tokens_removed:
                if tok not in stopword_list:
                    stopword_removal-=1
                    break;
            stopword_removal+=1;
            
            new_tokens = resolved_question
            for tok in tokens_common:
                new_tokens=new_tokens.replace(tok,'')
            if len(new_tokens.split())/len(resolved_question.split())>=0.6 and len(new_tokens.split())/len(resolved_question.split())<2:
            # if len(new_tokens.split())>=5 and len(new_tokens.split())<9:
                minor_change+=1
            
            
            
            return 1
        
        # return title, original_answer



stopword_list = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']



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






batch_list = [


[["edited_answers/user_split/Enamul/output_10.csv","edited_answers/user_split/Enamul/output_11.csv"],
 ['disagreements_resolved/manually/Enamul/batch_10(0-49).csv','disagreements_resolved/manually/Enamul/batch_10(50-99).csv']
 ],


[["edited_answers/user_split/Long/pair1/output_9.csv","edited_answers/user_split/Long/pair1/output_10.csv"],
['disagreements_resolved/manually/Long/pair1/batch_9(0-99).csv']
 ],

[["edited_answers/user_split/Long/pair2/output_13.csv","edited_answers/user_split/Long/pair2/output_14.csv"],
['disagreements_resolved/manually/Long/pair2/batch_13(0-19).csv','disagreements_resolved/manually/Long/pair2/batch_13(20-29).csv',
 'disagreements_resolved/manually/Long/pair2/batch_13(30-39).csv','disagreements_resolved/manually/Long/pair2/batch_13(40-49).csv',
 'disagreements_resolved/manually/Long/pair2/batch_13(50-80).csv','disagreements_resolved/manually/Long/pair2/batch_13(81-99).csv']
 ],
 
[["edited_answers/user_split/Shankar/pair1/output_11.csv","edited_answers/user_split/Shankar/pair1/output_12.csv"],
['disagreements_resolved/manually/Shankar/pair1/batch11(0-9).csv','disagreements_resolved/manually/Shankar/pair1/batch11(10-29).csv',
 'disagreements_resolved/manually/Shankar/pair1/batch11(30-49).csv','disagreements_resolved/manually/Shankar/pair1/batch11(50-79).csv',
 'disagreements_resolved/manually/Shankar/pair1/batch11(80-99).csv']
 ],

[["edited_answers/user_split/Shankar/pair2/output_14.csv","edited_answers/user_split/Shankar/pair2/output_15.csv"],
['disagreements_resolved/manually/Shankar/pair2/batch14(0-19).csv','disagreements_resolved/manually/Shankar/pair2/batch14(20-39).csv',
 'disagreements_resolved/manually/Shankar/pair2/batch14(40-69).csv','disagreements_resolved/manually/Shankar/pair2/batch14(70-99).csv']
 ],

[["edited_answers/user_split/Tan/pair1/output_8.csv","edited_answers/user_split/Tan/pair1/output_9.csv"],
['disagreements_resolved/manually/Tan/resolved/pair1/batch_8(0-9).csv','disagreements_resolved/manually/Tan/resolved/pair1/batch_8(10-29).csv',
 'disagreements_resolved/manually/Tan/resolved/pair1/batch_8(30-49).csv','disagreements_resolved/manually/Tan/resolved/pair1/batch_8(50-79).csv',
 'disagreements_resolved/manually/Tan/resolved/pair1/batch_8(80-99).csv']
 ],


[["edited_answers/user_split/Tan/pair2/output_12.csv","edited_answers/user_split/Tan/pair2/output_13.csv"],
['disagreements_resolved/manually/Tan/resolved/pair2/batch_12(0-19).csv','disagreements_resolved/manually/Tan/resolved/pair2/batch_12(20-39).csv',
 'disagreements_resolved/manually/Tan/resolved/pair2/batch_12(40-59).csv','disagreements_resolved/manually/Tan/resolved/pair2/batch_12(60-79).csv',
 'disagreements_resolved/manually/Tan/resolved/pair2/batch_12(80-99).csv']
 ],


[["edited_answers/user_split/Tiffany/output_7.csv","edited_answers/user_split/Tiffany/output_8.csv"],
['disagreements_resolved/manually/Tiffany/batch_7(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Enamul/pair1/output_15.csv","edited_answers/user_split_remaining/Enamul/pair1/output_16.csv"],
['disagreements_resolved/manually_remaining/Enamul/pair1/batch_15(0-29).csv','disagreements_resolved/manually_remaining/Enamul/pair1/batch_15(30-49).csv',
 'disagreements_resolved/manually_remaining/Enamul/pair1/batch_15(50-99).csv']
 ],

[["edited_answers/user_split_remaining/Enamul/pair2/output_16.csv","edited_answers/user_split_remaining/Enamul/pair2/output_17.csv"],
['disagreements_resolved/manually_remaining/Enamul/pair2/batch_16(0-19).csv','disagreements_resolved/manually_remaining/Enamul/pair2/batch_16(20-49).csv',
 'disagreements_resolved/manually_remaining/Enamul/pair2/batch_16(50-99).csv']
 ],


[["edited_answers/user_split_remaining/Enamul/pair3/output_17.csv","edited_answers/user_split_remaining/Enamul/pair3/output_18.csv"],
['disagreements_resolved/manually_remaining/Enamul/pair3/batch_17(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Long/pair1/output_18.csv","edited_answers/user_split_remaining/Long/pair1/output_19.csv"],
['disagreements_resolved/manually_remaining/Long/pair1/batch_18(0-99).csv']
 ],
 
[["edited_answers/user_split_remaining/Long/pair2/output_19.csv","edited_answers/user_split_remaining/Long/pair2/output_20.csv"],
['disagreements_resolved/manually_remaining/Long/pair2/batch_19(0-29).csv','disagreements_resolved/manually_remaining/Long/pair2/batch_19(30-39).csv',
 'disagreements_resolved/manually_remaining/Long/pair2/batch_19(40-49).csv','disagreements_resolved/manually_remaining/Long/pair2/batch_19(50-69).csv',
 'disagreements_resolved/manually_remaining/Long/pair2/batch_19(70-79).csv','disagreements_resolved/manually_remaining/Long/pair2/batch_19(80-99).csv']
 ],


[["edited_answers/user_split_remaining/Long/pair3/output_20.csv","edited_answers/user_split_remaining/Long/pair3/output_21.csv"],
['disagreements_resolved/manually_remaining/Long/pair3/batch_20(0-39).csv','disagreements_resolved/manually_remaining/Long/pair3/batch_20(40-79).csv',
 'disagreements_resolved/manually_remaining/Long/pair3/batch_20(80-89).csv','disagreements_resolved/manually_remaining/Long/pair3/batch_20(90-99).csv']
 ],


[["edited_answers/user_split_remaining/Shankar/pair1/output_21.csv","edited_answers/user_split_remaining/Shankar/pair1/output_22.csv"],
['disagreements_resolved/manually_remaining/Shankar/pair1/batch_21(0-19).csv','disagreements_resolved/manually_remaining/Shankar/pair1/batch_21(20-49).csv',
 'disagreements_resolved/manually_remaining/Shankar/pair1/batch_21(50-69).csv','disagreements_resolved/manually_remaining/Shankar/pair1/batch_21(70-99).csv']
 ],

[["edited_answers/user_split_remaining/Shankar/pair2/output_22.csv","edited_answers/user_split_remaining/Shankar/pair2/output_23.csv"],
['disagreements_resolved/manually_remaining/Shankar/pair2/batch_22(0-19).csv','disagreements_resolved/manually_remaining/Shankar/pair2/batch_22(20-49).csv',
 'disagreements_resolved/manually_remaining/Shankar/pair2/batch_22(50-79).csv','disagreements_resolved/manually_remaining/Shankar/pair2/batch_22(80-99).csv']
 ],

[["edited_answers/user_split_remaining/Shankar/pair3/output_23.csv","edited_answers/user_split_remaining/Shankar/pair3/output_24.csv"],
['disagreements_resolved/manually_remaining/Shankar/pair3/batch_23(0-19).csv','disagreements_resolved/manually_remaining/Shankar/pair3/batch_23(20-49).csv',
 'disagreements_resolved/manually_remaining/Shankar/pair3/batch_23(50-79).csv','disagreements_resolved/manually_remaining/Shankar/pair3/batch_23(80-99).csv']
 ],

[["edited_answers/user_split_remaining/Tan/pair1/output_24.csv","edited_answers/user_split_remaining/Tan/pair1/output_25.csv"],
['disagreements_resolved/manually_remaining/Tan/pair1/batch_24(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Tan/pair2/output_25.csv","edited_answers/user_split_remaining/Tan/pair2/output_26.csv"],
['disagreements_resolved/manually_remaining/Tan/pair2/batch_25(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Tan/pair3/output_26.csv","edited_answers/user_split_remaining/Tan/pair3/output_27.csv"],
['disagreements_resolved/manually_remaining/Tan/pair3/batch_26(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Tiffany/pair1/output_27.csv","edited_answers/user_split_remaining/Tiffany/pair1/output_28.csv"],
['disagreements_resolved/manually_remaining/Tiffany/pair1/batch_27(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Tiffany/pair2/output_28.csv","edited_answers/user_split_remaining/Tiffany/pair2/output_29.csv"],
['disagreements_resolved/manually_remaining/Tiffany/pair2/batch_28(0-99).csv']
 ],


[["edited_answers/user_split_remaining/Tiffany/pair3/output_29.csv","edited_answers/user_split_remaining/Tiffany/pair3/output_30.csv"],
['disagreements_resolved/manually_remaining/Tiffany/pair3/batch_29(0-99).csv']
 ],


[["edited_answers/user_split_last_900/Megh/output_30.csv","edited_answers/user_split_last_900/Megh/output_31.csv"],
['disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(0_29).csv','disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(30_39).csv',
 'disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(40_49).csv','disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(50_59).csv',
'disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(60_69).csv','disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(70_79).csv',
 'disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(80_89).csv','disagreements_resolved/Last_900/Megh/annotations_megh/batch_30(90_99).csv']
 ],


[["edited_answers/user_split_last_900/Shankar_n_Long/output_31.csv","edited_answers/user_split_last_900/Shankar_n_Long/output_32.csv"],
['disagreements_resolved/Last_900/Shankar_n_Long/batch_31(0-19).csv','disagreements_resolved/Last_900/Shankar_n_Long/batch_31(20-49).csv',
 'disagreements_resolved/Last_900/Shankar_n_Long/batch_31(50-69).csv','disagreements_resolved/Last_900/Shankar_n_Long/batch_31(70-89).csv',
 'disagreements_resolved/Last_900/Shankar_n_Long/batch_31(90-99).csv']
 ],



[["edited_answers/user_split_last_900/Tan_n_Tiffany/output_32.csv","edited_answers/user_split_last_900/Tan_n_Tiffany/output_33.csv"],
 ['disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(0-49).csv','disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(50_60).csv',
 'disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(61_75).csv','disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(76-99).csv']
 ]


]









total_question_change_count=0
total_dataset_count=0
stopword_removal=0
minor_change=0


for [file1,file2],file_resolved_batches in batch_list:


    #original files
    # file1 = "edited_answers/user_split_last_900/Tan_n_Tiffany/output_32.csv"
    # file2 = "edited_answers/user_split_last_900/Tan_n_Tiffany/output_33.csv"
    
    # output_dir = "disagreements_resolved/combined_final/batch_27.csv"
    
    # file_resolved_batches = ["disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(0-49).csv",
    #                          "disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(50_60).csv",
    #                          "disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(61_75).csv",
    #                          "disagreements_resolved/Last_900/Tan_n_Tiffany/batch_32(76-99).csv"
    #                            ]
    
    df1 = pd.read_csv(file1,encoding='utf8')
    df2 = pd.read_csv(file2,encoding='utf8')
    
    df_resolved = pd.read_csv(file_resolved_batches[0],encoding='utf8')
    
    assert len(df1) == len(df2)
    
    combined_output = pd.DataFrame(columns=df_resolved.columns)
    combined_output['title'] = []
    combined_output['original_answer'] = [] 
    
                   
    
    
    
    
    
    
    
    
    
    question_change_count=0
    for file in file_resolved_batches:
        df_resolved = pd.read_csv(file,encoding='utf8')
        for index, row in df_resolved.iterrows():
            title, original_answer = find_extra_info(row['img_path'],row['answer_resolution'],row['extractive_answer'])
            question_change_count+= find_question(row['img_path'],row['answer_resolution'],row['extractive_answer'],row['question'])
            # combined_output.loc[-1] = pd.concat([row, pd.Series([title,original_answer],index=['title','original_answer'])], axis=0)
            combined_output.loc[-1] = row       #if title and original_answer is available in the csv
            combined_output.index = combined_output.index + 1
            combined_output = combined_output.sort_index()
    
    #print(file1)
    #print(file2)
    #print("questions changed :", question_change_count)
    #print("questions same :", 300-question_change_count)
    total_question_change_count+=question_change_count
    total_dataset_count+=300
    # combined_output.to_csv(output_dir,encoding='utf8', index=False)



print("percentage of questions changed:",total_question_change_count/(total_dataset_count-len(filter_list)))
print("percentage of stopword removals", stopword_removal/(total_dataset_count-len(filter_list)))
print("percentage of minor changes", minor_change/(total_dataset_count-len(filter_list)))


    