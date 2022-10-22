# -*- coding: utf-8 -*-

import json




# gold_path = "data/test.json"
# predict_path = "output/test2/predictions.json"

target_path = "pew/bertqa_data/data.json" #generated summaries


target = {}
target['data']=[]


split = "test" #target summaries
dataset = json.load(open('pew/bertqa_data/%s.json' % split, 'r'))


# dataset[list(dataset.keys())[0]] #image_no, title, summary, question, abstractive_answer, extractive_answer


def find_start_n_answer(orig_answer_text,paragraph_text):
  answer_list = orig_answer_text.split(" ")
  max_match = 4
  start_index = -1
  end_index = -1
  for i in range(max_match,0,-1):
    if paragraph_text.find(" ".join(answer_list[:i])) != -1:
      start_index = paragraph_text.find(" ".join(answer_list[:i]))
      break

  for i in range(max_match,0,-1):
    if paragraph_text.rfind(" ".join(answer_list[-i:])) != -1:
      end_index = paragraph_text.rfind(" ".join(answer_list[-i:])) + len(" ".join(answer_list[-i:]))
      break
  return start_index, paragraph_text[start_index: end_index]


processed_data = []
for id in dataset:
  new_data = {}

  sample = dataset[id]
  image_no, question = sample[0], sample[3]
  summary, extractive_answer = sample[2], sample[5]
  start_index, extrative_answer = find_start_n_answer(extractive_answer, summary)
  if start_index == -1:
    continue

  new_data['context'] = summary
  new_data['qas'] = [{
      'question': question,
      'is_impossible': False,
      'id': id,
      'answers': [{
          'answer_start': start_index,
          'text': extrative_answer
      }]
  }]
  processed_data.append(new_data)
  
  

new_squad = {}
new_squad['version'] = "v2.0"
new_squad['data'] = []

new_squad_data = {
    "title": "training_data",
    "paragraphs": processed_data
}

new_squad['data'].append(new_squad_data)

  
  
with open(target_path, 'w') as outfile:
    json.dump(new_squad, outfile)
  
  
  
  