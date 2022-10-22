import pandas as pd
import nltk

# nltk.download("punkt")

COLUMNS = [
	"abstractive_answer",
	"extractive_answer",
	"abstractive_ocr_text",
	"abstractive_ocr_title_text",
	"abstractive_ocr_title_bboxes_text",
	"extractive_ocr_text",
	"extractive_ocr_title_text",
	"extractive_ocr_title_bboxes_text",
]

full_dataset_df = pd.read_csv("full_dataset.csv")

for col in COLUMNS:
	full_dataset_df[col] = full_dataset_df[col].apply(lambda text: len(nltk.word_tokenize(text)))

full_dataset_df[COLUMNS].describe().to_csv("token_stats.csv")
