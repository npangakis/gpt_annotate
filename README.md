# Introducing gpt_annotate
An easy-to-use Python package designed to streamline automated text annotation using LLMs for different tasks and datasets. All you need is an OpenAI API key, text samples you want to annotate, and a codebook (i.e., task-specific instructions) for the LLM.
* OpenAI API key 
	* Sign up for one here: https://platform.openai.com/account/api-keys
* text_to_annotate: 
	* A dataframe that includes one column for text samples and, if you are comparing the LLM output against humans, any number of one-hot-encoded category columns. The text column should be the first column in your data. We provide Python code (described below) that will automatically assist with the formatting of `text_to_annotate` to ensure accurate annotation.
* codebook: 
	* Task-specific instructions (as type string) to prompt the LLM to annotate the data. Like codebooks for qualitative content analysis, this should clearly describe the dataset, the type of task for the LLM, and, most importantly, delineate the categories of interest for the LLM to annotate. We provide Python code to standardize the beginning and ending of the codebook to ensure that the LLM understands that the task is annotation.
	* For example, the text of `codebook` could be: "You will be classifying text samples. Each text sample is a tweet. Classify each tweet on two dimensions: a) POLITICAL; b) PRESIDENT. For POLITICAL, label as 1 if the tweet is about politics; label as 0 if not. For PRESIDENT, label as 1 if the tweet refers to a past or present president, a candidate for president, or a presidential election; label as 0 if not. Classify the following text samples:"

To annotate your text data using gpt_annotate, we recommend following the sample code we provide in `sample_annotation_code.ipynb`.

As shown in `sample_annotation_code.ipynb`, annotating your text data with LLMs is as easy as 4 simple steps:
1. Import the required dependencies (including gpt_annotate.py).

```
import openai
import pandas as pd
import math
import time
import numpy as np
import tiktoken
#### Import main package: gpt_annotate.py
# Make sure that the .py file is in the same directory as the .ipynb file, or you provide the correct relative or absolute path to the .py file.
import gpt_annotate
```

2. Read in your codebook (i.e., task-specific instructions) and the text samples you want to annotate.

```
text_to_annotate = pd.read_csv("text_to_annotate.csv")
with open('codebook.txt', 'r') as file:
	codebook = file.read()
 ```
    
3. To ensure your data is in the right format, you must first run `gpt_annotate.prepare_data(text_to_annotate, codebook, key)`. If you are annotating text data without any human labels to compare against, change the default to `human_labels = False`. If you want to add standardized language to the beginning and end of your codebook to ensure that GPT will label your text samples, change the default to `prep_codebook = True`.
```
text_to_annotate = gpt_annotate.prepare_data(text_to_annotate, codebook, key)
```
4. If comparing LLM output to human labels, run `gpt_annotate.gpt_annotate(text_to_annotate, codebook, key)`. If only using gpt_annotate for prediction (i.e., no human labels to compare performance), run `gpt_annotate.gpt_annotate(text_to_annotate, codebook, key, human_labels = False)`. It’s as easy as that!
```
# Annotate the data (returns 4 outputs)
gpt_out_all, gpt_out_final, performance, incorrect =  gpt_annotate.gpt_annotate(text_to_annotate, codebook, key)
# Annotate the data (without human labels to compare against) (returns 2 outputs)
gpt_out_all, gpt_out_final =  gpt_annotate.gpt_annotate(text_to_annotate, codebook, key, human_labels = False)
```

Outputs:
1) `gpt_out_all`
  *   Raw outputs for every iteration.
2) `gpt_out_final`
  *   Annotation outputs after taking modal category answer and calculating consistency scores.
3) `performance`
  *   Accuracy, precision, recall, and f1.
4) `incorrect`
  *   Any incorrect classification or classification with less than 1.0 consistency.

Below we define the alternative parameters within `gpt_annotate()` to customize your annotation procedures.
* num_iterations:
	* Number of times to classify each text sample. Default is 3.
* model:
	* OpenAI GPT model, which is either `gpt-3.5-turbo` or `gpt-4`. Default is `gpt-4`.
* temperature: 
	* LLM temperature parameter (ranges 0 to 1), which indicates the degree of diversity to introduce into the model. Default is 0.6.
* batch_size:
	* Number of text samples to be annotated in each batch. Default is 10.
* human_labels: 
	* Boolean indicating whether `text_to_annotate` has human labels to compare LLM outputs to. 
* data_prep_warning: 
	* Boolean indicating whether to print data_prep_warning
* time_cost_warning: 
	* Boolean indicating whether to print time_cost_warning


Please email us (njpang@sas.upenn.edu) with any suggestions or problems encountered with the code.
