# BART Model
- To finetune the BART model for query-focussed chart summarisation, we used the [summarization example code from Hugging Face](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization)
- We finetuned the model for 2 tasks
	- Abstractive summarisation
	- Extractive summarisation

## How to Generate the Inputs
1. Download the dataset from https://drive.google.com/file/d/11nsA5km-8l03S5UoBmgU3D0Lc0zFcrrT/view?usp=sharing
2. Copy the `inputs/` folder into the dataset folder
3. `cd inputs/`
4. `python preprocessing.py`
	1. You can modify this file to change how the inputs are formatted

### Generating Token Statistics
1. `cd inputs/`
2. `python token_stats.py`
	1. For reference to set the input & output sequence length
	2. Generates the `inputs/token_stats.csv`

### Input Format
- Overall Format: `question <s> title <s> ocr <s> summary`
	- `title` & `summary` may be excluding depending on the set-up
- OCR Format: `ocr_text [a, b, c, d]`
	- `[a, b, c, d]` is omitted if no bounding boxes

## How to Run Models
### Setup
1. `git clone https://github.com/huggingface/transformers`
2. `cd transformers`
3. `pip install .`
4. `pip install -r examples/pytorch/summarization/requirements.txt`
5. `cd ..`

### Running the Models
```bash
# Abstractive
summary=abstractive model=abstractive_ocr sh run_bart_model.sh
summary=abstractive model=abstractive_ocr_title sh run_bart_model.sh
summary=abstractive model=abstractive_ocr_title_bboxes sh run_bart_model.sh

# Extractive
summary=extractive model=extractive_ocr sh run_bart_model.sh
summary=extractive model=extractive_ocr_title sh run_bart_model.sh
summary=extractive model=extractive_ocr_title_bboxes sh run_bart_model.sh
```
