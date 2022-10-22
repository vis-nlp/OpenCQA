# -*- coding: utf-8 -*-



import pandas as pd
import re


# automatically_resolved_dataframe= pd.DataFrame(columns=['img_path','summary','question','answer'])
# need_further_resolution1 = pd.DataFrame(columns=df1.columns)
# need_further_resolution2 = pd.DataFrame(columns=df2.columns)

no_match=[] #<30
partial=[] #>30,<90
full_match=[]#>90
exact_match=[]#100
glob_count=0

def calculate_score(row1,row2,img_path,summary,answer1,answer2,question1,question2):
    # answer1 = df1.loc[index,ans1]
    # answer2 = df2.loc[index,ans2]
    
    # question1 = df1.loc[index,ques1]
    # question2 = df2.loc[index,ques2]
    global glob_count
    glob_count+=1
    answer= ''
    question= ''
    global automatically_resolved_dataframe
    global need_further_resolution2
    global need_further_resolution1
    global no_match,partial,full_match
    
    if len(answer1) > len(answer2):
        match_words = [word for word in answer2.split() if word in answer1.split()]
        match_percent = len(match_words)/len(answer2.split())
        #if match_percent > 0.7 and question1 == question2: #70% of the words in the smaller sentence exists in the larger sentence and questions from both workers is same
        if match_percent > 0.9:
            # answer = answer1;
            # question = question1
            full_match.append(1)
        elif match_percent > 0.3:
            partial.append(1)
        else:
            no_match.append(1)
            # answer = None
    
    elif len(answer2) > len(answer1):
        match_words = [word for word in answer1.split() if word in answer2.split()]
        match_percent = len(match_words)/len(answer1.split())
        #if match_percent > 0.7 and question1 == question2:
        if match_percent > 0.9:
            # answer = answer2
            # question = question2
            full_match.append(1)
        elif match_percent > 0.3:
            partial.append(1)
        else:
            no_match.append(1)
            # answer = None
    
    elif len(answer1) == len(answer2):
        match_words = [word for word in answer2.split() if word in answer1.split()]
        match_percent = len(match_words)/len(answer2.split())
        #if question1 == question2:
        if match_percent == 1:
            # full_match.append(1)
            exact_match.append(1)
        elif match_percent > 0.9:
            full_match.append(1)
        elif match_percent > 0.3:
            partial.append(1)
        else:
            no_match.append(1)
            # answer = answer1
            # question = question1
        # else:
        #     answer = None
    
    

    # if answer == None:
    #         if row1['HITId'] not in need_further_resolution1['HITId'].values:
    #             need_further_resolution1.loc[len(need_further_resolution1)] = row1
            
    #         if row2['HITId'] not in need_further_resolution2['HITId'].values:
    #             need_further_resolution2.loc[len(need_further_resolution2)] = row2
             
    #         # for img in [row1['Input.img_path4'],row1['Input.img_path5'],row1['Input.img_path6']]:
    #         #     if img in automatically_resolved_dataframe['img_path'].values:
    #         #         automatically_resolved_dataframe = automatically_resolved_dataframe[automatically_resolved_dataframe.img_path != img]

                    
    # else: #if questions are equal and answers mostly equal 
    #         #if row1['HITId'] not in need_further_resolution1['HITId'].values and row2['HITId'] not in need_further_resolution2['HITId'].values:
    #             automatically_resolved_dataframe.loc[len(automatically_resolved_dataframe)] = [img_path,summary,question,answer]
                        
    
    
file1 = "edited_answers/user_split/Shankar/pair1/output_11.csv"
file2 = "edited_answers/user_split/Shankar/pair1/output_12.csv"

output_list = [["edited_answers/user_split/Enamul/output_10.csv","edited_answers/user_split/Enamul/output_11.csv"],
               ["edited_answers/user_split/Long/pair1/output_9.csv","edited_answers/user_split/Long/pair1/output_10.csv"],
               ["edited_answers/user_split/Long/pair2/output_13.csv","edited_answers/user_split/Long/pair2/output_14.csv"],
               ["edited_answers/user_split/Shankar/pair1/output_11.csv","edited_answers/user_split/Shankar/pair1/output_12.csv"],
               ["edited_answers/user_split/Shankar/pair2/output_14.csv","edited_answers/user_split/Shankar/pair2/output_15.csv"],
               ["edited_answers/user_split/Tan/pair1/output_8.csv","edited_answers/user_split/Tan/pair1/output_9.csv"],
               ["edited_answers/user_split/Tan/pair2/output_12.csv","edited_answers/user_split/Tan/pair2/output_13.csv"],
               ["edited_answers/user_split/Tiffany/output_7.csv","edited_answers/user_split/Tiffany/output_8.csv"],
               ["edited_answers/user_split_remaining/Enamul/pair1/output_15.csv","edited_answers/user_split_remaining/Enamul/pair1/output_16.csv"],
               ["edited_answers/user_split_remaining/Enamul/pair2/output_16.csv","edited_answers/user_split_remaining/Enamul/pair2/output_17.csv"],
               ["edited_answers/user_split_remaining/Enamul/pair3/output_17.csv","edited_answers/user_split_remaining/Enamul/pair3/output_18.csv"],
               ["edited_answers/user_split_remaining/Long/pair1/output_18.csv","edited_answers/user_split_remaining/Long/pair1/output_19.csv"],
               ["edited_answers/user_split_remaining/Long/pair2/output_19.csv","edited_answers/user_split_remaining/Long/pair2/output_20.csv"],
               ["edited_answers/user_split_remaining/Long/pair3/output_20.csv","edited_answers/user_split_remaining/Long/pair3/output_21.csv"],
               ["edited_answers/user_split_remaining/Shankar/pair1/output_21.csv","edited_answers/user_split_remaining/Shankar/pair1/output_22.csv"],
               ["edited_answers/user_split_remaining/Shankar/pair2/output_22.csv","edited_answers/user_split_remaining/Shankar/pair2/output_23.csv"],
               ["edited_answers/user_split_remaining/Shankar/pair3/output_23.csv","edited_answers/user_split_remaining/Shankar/pair3/output_24.csv"],
               ["edited_answers/user_split_remaining/Tan/pair1/output_24.csv","edited_answers/user_split_remaining/Tan/pair1/output_25.csv"],
               ["edited_answers/user_split_remaining/Tan/pair2/output_25.csv","edited_answers/user_split_remaining/Tan/pair2/output_26.csv"],
               ["edited_answers/user_split_remaining/Tan/pair3/output_26.csv","edited_answers/user_split_remaining/Tan/pair3/output_27.csv"],
               ["edited_answers/user_split_remaining/Tiffany/pair1/output_27.csv","edited_answers/user_split_remaining/Tiffany/pair1/output_28.csv"],
               ["edited_answers/user_split_remaining/Tiffany/pair2/output_28.csv","edited_answers/user_split_remaining/Tiffany/pair2/output_29.csv"],
               ["edited_answers/user_split_remaining/Tiffany/pair3/output_29.csv","edited_answers/user_split_remaining/Tiffany/pair3/output_30.csv"],
               ["edited_answers/user_split_last_900/Megh/output_30.csv","edited_answers/user_split_last_900/Megh/output_31.csv"],
               ["edited_answers/user_split_last_900/Shankar_n_Long/output_31.csv","edited_answers/user_split_last_900/Shankar_n_Long/output_32.csv"],
               ["edited_answers/user_split_last_900/Tan_n_Tiffany/output_32.csv","edited_answers/user_split_last_900/Tan_n_Tiffany/output_33.csv"]]


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



for files in output_list:
    
    df1 = pd.read_csv(files[0],encoding='utf8')
    df2 = pd.read_csv(files[1],encoding='utf8')


    assert len(df1) == len(df2)
    
    
    for row_no, row in df1.iterrows():
        # if df1.loc[index,'Input.img_path4'] == df2.loc[index,'Input.img_path1']:
            
            
        img_path = df1.loc[row_no,'Input.img_path4']
        if img_path in df2['Input.img_path1'].values and img_path not in filter_list:
            row_no2 = df2[df2['Input.img_path1'] == img_path].index.item()
            assert df1.loc[row_no,'Input.summary4'] == df2.loc[row_no2,'Input.summary1']
            summary = df1.loc[row_no,'Input.summary4']
            ans1 = df1.loc[row_no,'Answer.answer4']
            ans2 = df2.loc[row_no2,'Answer.answer1']
            ques1 = df1.loc[row_no,'Answer.question4']
            ques2 = df2.loc[row_no2,'Input.question1']
            row1 = df1.iloc[row_no]
            row2 = df2.iloc[row_no2]
            
            
            calculate_score(row1,row2,img_path,summary,ans1,ans2,ques1,ques2)
            
        # if df1.loc[index,'Input.img_path5'] == df2.loc[index,'Input.img_path2']:
            
            
        img_path = df1.loc[row_no,'Input.img_path5']
        if img_path in df2['Input.img_path2'].values and img_path not in filter_list:
            row_no2 = df2[df2['Input.img_path2'] == img_path].index.item()
            assert df1.loc[row_no,'Input.summary5'] == df2.loc[row_no2,'Input.summary2']
            summary = df1.loc[row_no,'Input.summary5']
            ans1 = df1.loc[row_no,'Answer.answer5']
            ans2 = df2.loc[row_no2,'Answer.answer2']
            ques1 = df1.loc[row_no,'Answer.question5']
            ques2 = df2.loc[row_no2,'Input.question2']
            row1 = df1.iloc[row_no]
            row2 = df2.iloc[row_no2]
            
            
            calculate_score(row1,row2,img_path,summary,ans1,ans2,ques1,ques2)
            
            
            
        # if df1.loc[index,'Input.img_path6'] == df2.loc[index,'Input.img_path3']:
                    
            
            
        img_path = df1.loc[row_no,'Input.img_path6']
        if img_path in df2['Input.img_path3'].values and img_path not in filter_list:
            row_no2 = df2[df2['Input.img_path3'] == img_path].index.item()
            assert df1.loc[row_no,'Input.summary6'] == df2.loc[row_no2,'Input.summary3']
            summary = df1.loc[row_no,'Input.summary6']
            ans1 = df1.loc[row_no,'Answer.answer6']
            ans2 = df2.loc[row_no2,'Answer.answer3']
            ques1 = df1.loc[row_no,'Answer.question6']
            ques2 = df2.loc[row_no2,'Input.question3']
            row1 = df1.iloc[row_no]
            row2 = df2.iloc[row_no2]
            
            calculate_score(row1,row2,img_path,summary,ans1,ans2,ques1,ques2)

total = len(full_match)+len(partial)+len(no_match)+len(exact_match)
print("exact match(=100%)",len(exact_match)/total * 100)
print("near-exact match(>90%)",len(full_match)/total * 100)
print("partial match(>30%)",len(partial)/total * 100)
print("bare match(<30%)",len(no_match)/total * 100)

# if len(automatically_resolved_dataframe) != 0:
#     automatically_resolved_dataframe.to_csv('disagreements_resolved/automatically/resolved_'+re.findall(r'\d+', file2)[0]+".csv",encoding='utf8', index=False)

# if len(need_further_resolution1) != 0:
#     need_further_resolution1.to_csv('unresolved_disagreements/unresolved_firsthalf_'+re.findall(r'\d+', file2)[0]+".csv",encoding='utf8', index=False)

# if len(need_further_resolution2) != 0:
#     need_further_resolution2.to_csv('unresolved_disagreements/unresolved_secondhalf_'+re.findall(r'\d+', file2)[0]+".csv",encoding='utf8', index=False)



  
    