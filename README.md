# OpenQA: Open-ended Question Answering with Charts

* Authors: Shankar Kantharaj, Xuan Long Do, Rixie Tiffany Ko Leong, Jia Qing Tan, Enamul Hoque, Shafiq Joty
* Paper Link: [OpenCQA](https://aclanthology.org/2022.emnlp-main.811/)


## I. OpenCQA Dataset
### 1. Dataset Structure

- Chart images: The OpenCQA images are available in the [chart_image](https://github.com/vis-nlp/OpenCQA/tree/main/chart_images) folder. 
- Chart metadata: The full annotations are available in the [etc/data(full_summary_article)](https://github.com/vis-nlp/OpenCQA/tree/main/etc/data(random_article_baseline)) folder. They are saved as (train/val/test)_extended.json files for the train/val/test split. Each sample in the json split has the following structure:
```
{	
   image_no:    [image_file,
				  title,
				  article,
				  summary,
				  question,
				  abstractive_answer,
				  extractive_answer
				],
	....
	....
	....
}
```
in which ```title, article, summary, question, abstractive_answer, extractive_answer``` stand for the chart title, the full article, the chart summary, the open-ended question, the question's abstractive answer (modified version of extractive answer), and the question's extractive answer.

- Bounding-boxes and OCRs: All the bounding boxes and OCRs are available in [bboxes](https://github.com/vis-nlp/OpenCQA/tree/main/bboxes). Each chart image has a list of bounding boxes. Each bounding box sample has the OCR ```sentence```, and bbox format: ```[x_center, y_center, width, height]```.

- Note: Chart image ```X.png``` has the bounding box data named ```X.json```.

### 2. Finetuning Details
- Experimental types:
  - Extractive experiments: Use ```extractive_answer``` as the ground-truth answer.
  - Generative experiments: Use ```anstractive_answer``` as the ground-truth answer.
- Experimental setups:
  - With Article: 
    - Input: ```question + title + OCRs + article```.
    - Output: ```extractive_/abstractive_answer```.
  - With Summary: 
    - Input: ```question + title + OCRs + summary```.
    - Output: ```extractive_/abstractive_answer```.
  - Without Summary:
    - Input: ```question + title + OCRs```.
    - Output: ```extractive_/abstractive_answer```.

 
## II. Models

### 1. VL-T5
Please refer to [VL-T5](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/VLT5)

### 2. T5 
Please refer to [T5](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/T5)

### 3. BART
Please refer to [BART](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/bart)

### 4. BERTQA 
Please refer to [BERTQA](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/bertqa)

### 5. DOC2GRND 
Please refer to [DOC2GRND](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/doc2grnd)

### 6. ELECTRA
Please refer to [ELECTRA](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/electra)

### 7. GPT2 
Please refer to [GPT2](https://github.com/vis-nlp/OpenCQA/tree/main/baseline_models/gpt2)

## III. Evaluations
The metrics used in this work are listed in [evaluation_metrics](https://github.com/vis-nlp/OpenCQA/tree/main/evaluation_metrics). For each metric, we have ```steps.txt``` which presents the steps to setup and run the metric.

# Contact
If you have any questions about this work, please contact **Enamul Hoque** using the following email address: **enamulh@yorku.ca**.
 

# Reference
Please cite our paper if you use our models or dataset in your research. 

```
@inproceedings{kantharaj-etal-2022-opencqa,
    title = "{O}pen{CQA}: Open-ended Question Answering with Charts",
    author = "Kantharaj, Shankar  and
      Do, Xuan Long  and
      Leong, Rixie Tiffany  and
      Tan, Jia Qing  and
      Hoque, Enamul  and
      Joty, Shafiq",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.811",
    pages = "11817--11837",
    abstract = "Charts are very popular to analyze data and convey important insights. People often analyze visualizations to answer open-ended questions that require explanatory answers. Answering such questions are often difficult and time-consuming as it requires a lot of cognitive and perceptual efforts. To address this challenge, we introduce a new task called OpenCQA, where the goal is to answer an open-ended question about a chart with descriptive texts. We present the annotation process and an in-depth analysis of our dataset. We implement and evaluate a set of baselines under three practical settings. In the first setting, a chart and the accompanying article is provided as input to the model. The second setting provides only the relevant paragraph(s) to the chart instead of the entire article, whereas the third setting requires the model to generate an answer solely based on the chart. Our analysis of the results show that the top performing models generally produce fluent and coherent text while they struggle to perform complex logical and arithmetic reasoning.",
}
```