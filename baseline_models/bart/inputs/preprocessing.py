import pandas as pd
import json

# COLUMNS = [
#  	"image_no",
#  	"title",
#  	"summary",
#  	"question",
#  	"abstractive_answer",
#  	"extractive_answer",
# ]

COLUMNS = [
 	"image_no",
 	"title",
    "article",
 	"summary",
 	"question",
 	"abstractive_answer",
 	"extractive_answer"
]

# Data
train_df = pd.read_json("../../../etc/data(full_summary_article)/train_extended.json", orient="index").rename(columns={i:column for i, column in enumerate(COLUMNS)})
val_df = pd.read_json("../../../etc/data(full_summary_article)/val_extended.json", orient="index").rename(columns={i:column for i, column in enumerate(COLUMNS)})
test_df = pd.read_json("../../../etc/data(full_summary_article)/test_extended.json", orient="index").rename(columns={i:column for i, column in enumerate(COLUMNS)})
full_dataset_df = pd.read_json("../../../etc/data(full_summary_article)/full_dataset_extended.json", orient="index").rename(columns={i:column for i, column in enumerate(COLUMNS)})

dfs = {
	"train": train_df,
	"val": val_df,
	"test": test_df,
	"full_dataset": full_dataset_df,
}



# Preprocessing
def get_bboxes(image_no):
	index = image_no.split(".")[0]

	with open(f"../../../../bboxes/{index}.json") as f:
		return f.read()




def get_ocr(bboxes, include_bboxes=True):
	items = json.loads(bboxes)

	if include_bboxes:
		sentences = [f"{item['sentence'].strip()} {item['bounding_box']} " for item in items]
	else:
		sentences = [item["sentence"] for item in items]
	
	return "| ".join(sentences).strip()

def get_text(row, include_question=True, include_title=False, include_ocr=True, include_bboxes=False, include_summary=False):
    question = row["question"]
    title = row["title"]
    # summary = row["summary"]
    summary = row["article"]
    bboxes = row["bboxes"]
    
    text_elements = []
    
    if include_question:
    		text_elements.append(f"{question}")
    
    if include_title:
    		text_elements.append(f"{title}")
    
    if include_ocr:
    		text_elements.append(f"{get_ocr(bboxes, include_bboxes)}")
    
    if include_summary:
    		text_elements.append(f"{summary}")
    
    return " <s> ".join(text_elements)


for title, df in dfs.items():
    df.index.rename("id", inplace=True)
    
    df["bboxes"] = df["image_no"].apply(get_bboxes)
    df["abstractive_text"] = df.apply(lambda row: get_text(row, include_ocr=False), axis=1)
    df["abstractive_ocr_text"] = df.apply(lambda row: get_text(row), axis=1)
    df["abstractive_ocr_title_text"] = df.apply(lambda row: get_text(row, include_title=True), axis=1)
    df["abstractive_ocr_title_bboxes_text"] = df.apply(lambda row: get_text(row, include_title=True, include_bboxes=True), axis=1)
    
    df["extractive_text"] = df.apply(lambda row: get_text(row, include_summary=True, include_ocr=False), axis=1)
    df["extractive_ocr_text"] = df.apply(lambda row: get_text(row, include_summary=True), axis=1)
    df["extractive_ocr_title_text"] = df.apply(lambda row: get_text(row, include_summary=True, include_title=True), axis=1)
    df["extractive_ocr_title_bboxes_text"] = df.apply(lambda row: get_text(row, include_summary=True, include_title=True, include_bboxes=True), axis=1)
    
    df.to_csv(f"{title}.csv")
