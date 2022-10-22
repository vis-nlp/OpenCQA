from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import pandas as pd
import csv

folder = "output_T5_base_QFCS/checkpoint-1000/"

tokenizer = T5Tokenizer.from_pretrained(folder)
config = T5Config.from_json_file(folder + 'config.json')
model = T5ForConditionalGeneration.from_pretrained(folder + 'pytorch_model.bin', config=config)

test_text = list(pd.read_csv("Data/test.csv")["Text"])
test_img = list(pd.read_csv("Data/test.csv")["Image"])
test_summary = list(pd.read_csv("Data/test.csv")["Summary"])

print(len(test_text))
model_output_summary = []
for i in range(0, len(test_text), 4):
  batch = tokenizer(test_text[i:i+4], return_tensors='pt', max_length=1024, padding="max_length", truncation=True)
  generated_ids = model.generate(batch['input_ids'], attention_mask=batch["attention_mask"], max_length=250, num_beams=4)
  model_output_summary.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
  print("Finish with batch number: "+str(i))

import csv
with open('output/output_to_text.csv', 'w') as file:
  header_for_write = ["Image", "Text", "Summary", "Model Output"]
  writer = csv.writer(file)
  writer.writerow(header_for_write)
  for i in range(len(model_output_summary)):
    writer.writerow([test_img[i], test_text[i], test_summary[i], model_output_summary[i]]) 

model_output = []
test = []

with open('output/output_to_text.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        test.append(row[2])
        model_output.append(row[3])

from sacremoses import MosesTokenizer, MosesDetokenizer

mt = MosesTokenizer(lang="en")
md = MosesDetokenizer(lang="en")
def detokenize(sent):
    tokens = mt.tokenize(sent)
    return md.detokenize(tokens)

model_output = list(map(detokenize, model_output))
test = list(map(detokenize, test))

with open('output/model_out.txt', 'a') as the_file:
  for line in model_output[1:]:
    the_file.write(line + '\n')

with open('output/baseline_out.txt', 'a') as the_file:
  for line in test[1:]:
    the_file.write(line + '\n')