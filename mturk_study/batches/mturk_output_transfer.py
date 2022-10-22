# -*- coding: utf-8 -*-



import pandas as pd
import os
from configparser import ConfigParser




config = ConfigParser()
config.read('config.ini')





batch_no = int(config.get('DEFAULT', 'batch_no')) #mturk batch number






output_csv = "edited_questions/output_" + str(batch_no) + ".csv"
input_csv = "inputs/input_" + str(batch_no+1) + ".csv"


# output_csv="edited_answers/output_5.csv"
# input_csv="edited_questions/input_5.csv"





input_template = pd.read_csv("inputs/input_template.csv",encoding='utf8')


output_table = pd.read_csv(output_csv,encoding='utf8')







for index,row in output_table.iterrows():
    input_template.loc[index ,'img_path1'] = output_table.loc[index ,'Input.img_path4']
    input_template.loc[index ,'title1'] = output_table.loc[index ,'Input.title4']
    input_template.loc[index ,'summary1'] = output_table.loc[index ,'Input.summary4']
    input_template.loc[index ,'question1'] = output_table.loc[index ,'Answer.question4']
    input_template.loc[index ,'img_path2'] = output_table.loc[index ,'Input.img_path5']
    input_template.loc[index ,'title2'] = output_table.loc[index ,'Input.title5']
    input_template.loc[index ,'summary2'] = output_table.loc[index ,'Input.summary5']
    input_template.loc[index ,'question2'] = output_table.loc[index ,'Answer.question5']
    input_template.loc[index ,'img_path3'] = output_table.loc[index ,'Input.img_path6']
    input_template.loc[index ,'title3'] = output_table.loc[index ,'Input.title6']
    input_template.loc[index ,'summary3'] = output_table.loc[index ,'Input.summary6']
    input_template.loc[index ,'question3'] = output_table.loc[index ,'Answer.question6']





input_template.to_csv(input_csv,encoding='utf8', index=False)


config.set('DEFAULT', 'batch_no', str(batch_no+1))



with open('config.ini', 'w') as f:
    config.write(f)









