# -*- coding: utf-8 -*-
"""
## Overview

Given a codebook (.txt) and a dataset (.csv) that has one text column and any number of category columns as binary indicators, the main function (`gpt_annotate`) annotates
all the samples using an OpenAI GPT model (ChatGPT or GPT-4) and calculates performance metrics (if there are provided human labels). Before running `gpt_annotate`,
users should run `prepare_data` to ensure that their data is in the correct format.

Flow of `gpt_annotate`:
*   1) Based on a provided codebook, the function uses an OpenAI GPT model to annotate every text sample per iteration, which is a parameter set by user.
*   2) The function reduces the annotation output down to the modal annotation category across iterations for each category. At this stage,
       the function adds a consistency score for each annotation across iterations.
*   3) If provided human labels, the function determines, for every category, whether the annotation is correct (by comparing to the human label),
        then also adds whether it is a true positive, false positive, true negative, or false negative.
*   4) Finally, if provided human labels, the function calculates performance metrics (accuracy, precision, recall, and f1) for every category.

The main function (`gpt_annotate`) returns four .csv's, if human labels are provided.
If no human labels are provided, the main function only returns 1 and 2 listed below.
*   1) `gpt_out_all_iterations.csv`
  *   Raw outputs for every iteration.
*   2) `gpt_out_final.csv`
  *   Annotation outputs after taking modal category answer and calculating consistency scores.
*   3) `performance_metrics.csv`
  *   Accuracy, precision, recall, and f1.
*   4) `incorrect.csv`
  *   Any incorrect classification or classification with less than 1.0 consistency.

Our code aims to streamline automated text annotation for different datasets and numbers of categories.

"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("openai")
install("pandas")
install("numpy")
install("tiktoken")

import openai
import pandas as pd
import math
import time
import numpy as np
import tiktoken
from openai import OpenAI
import os


def prepare_data(text_to_annotate, codebook, key,
                 prep_codebook = False, human_labels = True, no_print_preview = False):

  """
  This function ensures that the data is in the correct format for LLM annotation. 
  If the data fails any of the requirements, returns the original input dataframe.

  text_to_annotate: 
      Data that will be prepared for analysis. Should include a column with text to annotate, and, if human_labels = True, the human labels.
  codebook: 
      String detailing the task-specific instructions.
  key:
    OpenAI API key
  prep_codebook: 
      boolean indicating whether to standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples.
  human_labels: 
      boolean indicating whether text_to_annotate has human labels to compare LLM outputs to. 
  no_print_preview:
    Does not print preview of user's data after preparing.

  Returns:
    Updated dataframe (text_to_annotate) and codebook (if prep_codebook = True) that are ready to be used for annotation using gpt_annotate.
  """

  # Check if text_to_annotate is a dataframe
  if not isinstance(text_to_annotate, pd.DataFrame):
    print("Error: text_to_annotate must be pd.DataFrame.")
    return text_to_annotate

  # Make copy of input data
  original_df = text_to_annotate.copy()

  # set OpenAI key
  openai.api_key = key

  # Standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples 
  if prep_codebook == True:
    codebook = prepare_codebook(codebook)

  # Add unique_id column to text_to_annotate
  text_to_annotate = text_to_annotate \
                    .reset_index() \
                    .rename(columns={'index':'unique_id'})

  ##### Minor Cleaning
  # Drop any Unnamed columns
  if any('Unnamed' in col for col in text_to_annotate.columns):
    text_to_annotate = text_to_annotate.drop(text_to_annotate.filter(like='Unnamed').columns, axis=1)
  # Drop any NA values
  text_to_annotate = text_to_annotate.dropna()

  ########## Confirming data is in correct format
  ##### 1) Check whether second column is string
  # rename second column to be 'text'
  text_to_annotate.columns.values[1] = 'text'
  # check whether second column is string
  if not (text_to_annotate.iloc[:, 1].dtype == 'string' or text_to_annotate.iloc[:, 1].dtype == 'object'):
    print("ERROR: Second column should be the text that you want to annotate.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df
  
  ##### 2) If human_labels == False, there should only be 2 columns
  if human_labels == False and len(text_to_annotate.columns) != 2:
    print("ERROR: You have set human_labels = False, which means you should only have two columns in your data.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df

  ##### 3) If human_labels == True, there should be more than 2 columns
  if human_labels == True and len(text_to_annotate.columns) < 3:
    print("ERROR: You have set human_labels = True (default value), which means you should have more than 2 columns in your data.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df


  ##### 4) If human_labels == True, check if columns 3 through end of df are binary indicators
  if human_labels == True:
    # iterate through each column (after unique_id and text) to confirm that the values are only 0's and 1's
    for col in range(2, len(text_to_annotate.columns)):
      if set(text_to_annotate.iloc[:,col].unique()) != {0, 1}:
        print("ERROR: All category columns must be one-hot encoded (i.e., represent numeric or categorical variables as binary vectors of type integer).")
        print("Make sure that all category columns only contain 1's and 0's.")
        print("")
        print("Your data:")
        print(text_to_annotate.head())
        print("")
        print("Sample data format:")
        error_message(human_labels)
        return original_df
 
  ##### 5) Add llm_query column that includes a unique ID identifier per text sample
  text_to_annotate['llm_query'] = text_to_annotate.apply(lambda x: str(x['unique_id']) + " " + str(x['text']) + "\n", axis=1)

  ##### 6) Make sure category names in codebook exactly match category names in text_to_annotate
  # extract category names from codebook
  if human_labels:
    col_names_codebook = get_classification_categories(codebook, key)
    # get category names in text_to_annotate
    df_cols = text_to_annotate.columns.values.tolist()
    # remove 'unique_id', 'text' and 'llm_query' from columns
    col_names = [col for col in df_cols if col not in ['unique_id','text', 'llm_query']]
    ### Check whether categories are the same in codebook and text_to_annotate
    if [col for col in col_names] != col_names_codebook:
      print("ERROR: Column names in codebook and text_to_annotate do not match exactly. Please note that order and capitalization matters.")
      print("Change order/spelling in codebook or text_to_annotate.")
      print("")
      print("Exact order and spelling of category names in text_to_annotate: ", col_names)
      print("Exact order and spelling of category names in codebook: ", col_names_codebook)
      return original_df
  else:
    col_names = get_classification_categories(codebook, key)

  ##### Confirm correct categories with user
  # Print annotation categories
  print("")
  print("Categories to annotate:")
  for index, item in enumerate(col_names, start=1):
    print(f"{index}) {item}")
  print("")
  if no_print_preview == False:
    waiting_response = True
    while waiting_response:
      # Confirm annotation categories
      input_response = input("Above are the categories you are annotating. Is this correct? (Options: Y or N) ")
      input_response = str(input_response).lower()
      if input_response == "y" or input_response == "yes":
        print("")
        print("Data is ready to be annotated using gpt_annotate()!")
        print("")
        print("Glimpse of your data:")
        # print preview of data
        print("Shape of data: ", text_to_annotate.shape)
        print(text_to_annotate.head())
        return text_to_annotate
      elif input_response == "n" or input_response == "no":
        print("")
        print("Adjust your codebook to clearly indicate the names of the categories you would like to annotate.")
        return original_df
      else:
        print("Please input Y or N.")
  else:
    return text_to_annotate


def gpt_annotate(text_to_annotate, codebook, key, 
                 num_iterations = 3, model = "gpt-4", temperature = 0.6, batch_size = 10,
                 human_labels = True,  data_prep_warning = True, 
                 time_cost_warning = True):
  """
  Loop over the text_to_annotate rows in batches and classify each text sample in each batch for multiple iterations. 
  Store outputs in a csv. Function is calculated in batches in case of crash.

  text_to_annotate:
    Input data that will be annotated.
  codebook:
    String detailing the task-specific instructions.
  key:
    OpenAI API key.
  num_iterations:
    Number of times to classify each text sample.
  model:
    OpenAI GPT model, which is either gpt-3.5-turbo or gpt-4
  temperature: 
    LLM temperature parameter (ranges 0 to 1), which indicates the degree of diversity to introduce into the model.
  batch_size:
    number of text samples to be annotated in each batch.
  human_labels: 
    boolean indicating whether text_to_annotate has human labels to compare LLM outputs to. 
  data_prep_warning: 
    boolean indicating whether to print data_prep_warning
  time_cost_warning: 
    boolean indicating whether to print time_cost_warning



  Returns:
    gpt_annotate returns the four .csv's below, if human labels are provided. If no human labels are provided, 
    gpt_annotate only returns gpt_out_all_iterations.csv and gpt_out_final.csv
    
    1) `gpt_out_all_iterations.csv`
        Raw outputs for every iteration.
    2) `gpt_out_final.csv`
        Annotation outputs after taking modal category answer and calculating consistency scores.
    3) `performance_metrics.csv`
        Accuracy, precision, recall, and f1.
    4) `incorrect.csv`
        Any incorrect classification or classification with less than 1.0 consistency.

  """

  
  from openai import OpenAI
  
  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)

  # set OpenAI key
  openai.api_key = key

  # Double check that user has confirmed format of the data
  if data_prep_warning:
    waiting_response = True
    while waiting_response:
      input_response = input("Have you successfully run prepare_data() to ensure text_to_annotate is in correct format? (Options: Y or N) ")
      input_response = str(input_response).lower()
      if input_response == "y" or input_response == "yes":
        # If user has run prepare_data(), confirm that data is in correct format
        # first, check if first column is title "unique_id"
        if text_to_annotate.columns[0] == 'unique_id' and text_to_annotate.columns[-1] == 'llm_query':
            try:
              text_to_annotate_copy = text_to_annotate.iloc[:, 1:-1]
            except pd.core.indexing.IndexingError:
              print("")
              print("ERROR: Run prepare_data(text_to_annotate, codebook, key) before running gpt_annotate(text_to_annotate, codebook, key).")
            text_to_annotate_copy = prepare_data(text_to_annotate_copy, codebook, key, prep_codebook = False, human_labels = human_labels, no_print_preview = True)
            # if there was an error, exit gpt_annotate
            if text_to_annotate_copy.columns[0] != "unique_id":
              if human_labels:
                return None, None, None, None
              elif human_labels == False:
                return None, None
        else:
            print("ERROR: First column should be title 'unique_id' and last column should be titled 'llm_query'")
            print("Try running prepare_data() again")
            print("")
            print("Your data:")
            print(text_to_annotate.head())
            print("")
            print("Sample data format:")
            if human_labels:
                return None, None, None, None
            elif human_labels == False:
                return None, None
        waiting_response = False
      elif input_response == "n" or input_response == "no":
        print("")
        print("Run prepare_data(text_to_annotate, codebook, key) before running gpt_annotate(text_to_annotate, codebook, key).")
        if human_labels:
            return None, None, None, None
        elif human_labels == False:
            return None, None
      else:
        print("Please input Y or N.")

  # df to store results
  out = pd.DataFrame()

  # Determine number of batches
  num_rows = len(text_to_annotate)
  num_batches = math.floor(num_rows/batch_size)
  num_iterations = num_iterations

  # Add categories to classify
  col_names = ["unique_id"] + text_to_annotate.columns.values.tolist()[2:-1]
  if human_labels == False:
    col_names = get_classification_categories(codebook, key)
    col_names = ["unique_id"] + col_names

  ### Nested for loop for main function
  # Iterate over number of classification iterations
  for j in range(num_iterations):
    # Iterate over number of batches
    for i in range(num_batches):

      # Based on batch, determine starting row and end row
      start_row = i*batch_size
      end_row = (i+1)*batch_size

      # Extract the text samples to annotate
      llm_query = text_to_annotate['llm_query'][start_row:end_row].str.cat(sep=' ')

      # Start while loop in case GPT fails to annotate a batch
      need_response = True
      while need_response:
        fails = 0
        # confirm time and cost with user before annotating data
        if fails == 0 and j == 0 and i == 0 and time_cost_warning:
          quit = estimate_time_cost(text_to_annotate, codebook, llm_query, model, num_iterations, num_batches, batch_size, col_names[1:])
          if quit and human_labels:
            return None, None, None, None
          elif quit and human_labels == False:
            return None, None
        # if GPT fails to annotate a batch 3 times, skip the batch
        while(fails < 3):
          try:
            # Set temperature
            temperature = temperature
            # annotate the data by prompting GPT
            response = get_response(codebook, llm_query, model, temperature, key)
            # parse GPT's response into a clean dataframe
            text_df_out = parse_text(response, col_names)
            break
          except:
            fails += 1
            pass
        if (',' in response.choices[0].message.content  or '|' in response.choices[0].message.content):
          need_response = False 

      # update iteration
      text_df_out['iteration'] = j+1

      # add iteration annotation results to output df
      out = pd.concat([out, text_df_out])
      time.sleep(.5)
    # print status report  
    print("iteration: ", j+1, "completed")

  # Convert unique_id col to numeric
  out['unique_id'] = pd.to_numeric(out['unique_id'])

  # Combine input df (i.e., df with text column and true category labels)
  out_all = pd.merge(text_to_annotate, out, how="inner", on="unique_id")

  # replace any NA values with 0's
  out_all.replace('', np.nan, inplace=True)
  out_all.replace('-', np.nan, inplace=True)
  out_all.fillna(0, inplace=True)

  ##### output 1: full annotation results
  out_all.to_csv('gpt_out_all_iterations.csv',index=False)

  # calculate modal label and consistency score
  out_mode = get_mode_and_consistency(out_all, col_names,num_iterations,human_labels)

  if human_labels == True:
    # evaluate classification per category
    num_categories = len(col_names) - 1 # account for unique_id
    for label in range(0, num_categories):
      out_final = evaluate_classification(out_mode, label, num_categories)

    ##### output 2: final annotation results with modal label and consistency score
    out_final.to_csv('gpt_out_final.csv',index=False)

    # calculate performance metrics
    performance = performance_metrics(col_names, out_final)

    ##### output 3: performance metrics
    performance.to_csv('performance_metrics.csv',index=False)
      
    # Determine incorrect classifications and classifications with less than 1.0 consistency
    incorrect = filter_incorrect(out_final)

    ##### output 4: Incorrect classifications 
    incorrect.to_csv('incorrect_sub.csv',index=False)
    return out_all, out_final, performance, incorrect

  else:
    # if not assessing performance against human annotators, then only save out_mode
    out_final = out_mode.copy()
    out_final.to_csv('gpt_out_final.csv',index=False)
    return out_all, out_final

########### Helper Functions

def prepare_codebook(codebook):
  """
  Standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples. 

  codebook: 
      String detailing the task-specific instructions.

  Returns:
    Updated codebook ready for annotation.
  """
  beginning = "Use this codebook for text classification. Return your classifications in a table with one column for text number (the number preceding each text sample) and a column for each label. Use a csv format. "
  end = " Classify the following text samples:"
  return beginning + codebook + end

def error_message(human_labels = True):
  """
  Prints sample data format if error.
  
  human_labels: 
      boolean indicating whether text_to_annotate has human labels to compare LLM outputs to.
  """
  if human_labels == True:
    toy_data = {
      'unique_id': [0, 1, 2, 3, 4],
      'text': ['sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate'],
      'category_1': [1, 0, 1, 0, 1],
      'category_2': [0, 1, 1, 0, 1]
      }
    toy_data = pd.DataFrame(toy_data)
    print(toy_data)
  else:
    toy_data = {
        'unique_id': [0, 1, 2, 3, 4],
      'text': ['sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate'],
      }
    toy_data = pd.DataFrame(toy_data)
    print(toy_data)

def get_response(codebook, llm_query, model, temperature, key):
  """
  Function to query OpenAI's API and get an LLM response.

  Codebook: 
    String detailing the task-specific instructions
  llm_query: 
    The text samples to append to the task-specific instructions
  Model: 
    gpt-3.5-turbo (Chat-GPT) or GPT-4
  Temperature: 
    LLM temperature parameter (ranges 0 to 1)

  Returns:
    LLM output.
  """

  from openai import OpenAI
  
  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)
  
  # max tokens for gpt-3.5-turbo is 4096 and max for gpt-4 is 8000
  if model == "gpt-3.5-turbo":
    max_tokens = 500
  elif model == "gpt-4":
    max_tokens = 1000

  # Create function to llm_query GPT
  # Create function to llm_query GPT
  response = client.chat.completions.create(
    model=model, # chatgpt: gpt-3.5-turbo # gpt-4: gpt-4
    messages=[
      {"role": "user", "content": codebook + llm_query}],
    temperature=temperature, # ChatGPT default is 0.7 (set lower reduce variation across queries)
    max_tokens = max_tokens,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  return response

def get_classification_categories(codebook, key):
  """
  Function that extracts what GPT will label each annotation category to ensure a match with text_to_annotate.
  Order and exact spelling matter. Main function will not work if these do not match perfectly.

  Codebook: 
    String detailing the task-specific instructions

  Returns:
    Categories to be annotated, as specified in the codebook.
  """

  # llm_query to ask GPT for categories from codebook
  instructions = "Below I am going to provide you with two sets of instructions (Part 1 and Part 2). The first part will contain detailed instructions for a text classification project. The second part will ask you to identify information about part 1. Prioritize the instructions from Part 2. Part 1:"
  llm_query = "Part 2: I've provided a codebook in the previous sentences. Please print the categories in the order you will classify them. Ignore every other task that I described in the codebook.   I only want to know the categories. Do not include any text numbers or any annotations in your response. Do not include any language like 'The categories to be identified are:'. Only include the names of the categories you are identifying. : "

  # Set temperature to 0 to make model deterministic
  temperature = 0
  # set model to GPT-4
  model = "gpt-4"

  
  from openai import OpenAI
  
  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)

  ### Get GPT response and clean response
  response = get_response(codebook, llm_query, model, temperature, key)
  text = response.choices[0].message.content
  text_split = text.split('\n')
  text_out = text_split[0]
  # text_out_list is final output of categories as a list
  codebook_columns = text_out.split(', ')

  return codebook_columns

def parse_text(response, headers):
  """
  This function converts GPT's output to a dataframe. GPT sometimes returns the output in different formats.
  Because there is variability in GPT outputs, this function handles different variations in possible outputs.

  response:
    LLM output
  headers:
    column names for text_to_annotate dataframe

  Returns:
    GPT output as a cleaned dataframe.

  """
  try:
    text = response.choices[0].message.content
    text_split = text.split('\n')

    if any(':' in element for element in text_split):
      text_split_split = [item.split(":") for item in text_split]

    if ',' in text:
      text_split_out = [row for row in text_split if (',') in row]
      text_split_split = [text.split(',') for text in text_split_out]
    if '|' in text:
      text_split_out = [row for row in text_split if ('|') in row]
      text_split_split = [text.split('|') for text in text_split_out]

    for row in text_split_split:
      if '' in row:
        row.remove('')
      if '' in row:
        row.remove('')
      if ' ' in row:
        row.remove(' ')

    text_df = pd.DataFrame(text_split_split)
    text_df_out = pd.DataFrame(text_df.values[1:], columns=headers)
    text_df_out = text_df_out[pd.to_numeric(text_df_out.iloc[:,1], errors='coerce').notnull()]
  except Exception as e:
        print("ERROR: GPT output not in specified categories. Make your codebook clearer to indicate what the output format should be.")
        print("Try running prepare_data(text_to_annotate, codebook, key, prep_codebook = True")
        print("")
  return text_df_out

def get_mode_and_consistency(df, col_names, num_iterations, human_labels):
  """
  This function calculates the modal label across iterations and calculates the 
  LLMs consistency score across label annotations.

  df:
    Input dataframe (text_to_annotate)
  col_names:
    Category names to be annotated
  num_iterations:
    number of iterations in gpt_annotate
  
  Returns:
    Modal label across iterations and consistency score for every text annotation.

  """

  # Drop unique_id column in list of category names
  categories_names = col_names[1:]
  # Change names to add a 'y' at end to match the output df, if human_labels == True
  if human_labels == True:
    categories_names = [name + "_y" for name in categories_names]

  ##### Calculate modal label classification across iterations
  # Convert dataframe to numeric
  df_numeric = df.apply(pd.to_numeric, errors='coerce')
  # Group by unique_id
  grouped = df_numeric.groupby('unique_id')
  # Calculate modal label classification per unique id (.iloc[0] means take the first value if there are ties)
  modal_values = grouped[categories_names].apply(lambda x: x.mode().iloc[0])
  # Create consistency score by calculating the number of times the modal answer appears per iteration
  consistency = grouped[categories_names].apply(lambda x: (x.mode().iloc[0] == x).sum()/num_iterations)

  ##### Data cleaning for new dfs related to consistency scores
  # add the string 'consistency_' to the column names of the consistency df
  consistency.columns = ['consistency_' + col for col in consistency.columns]
  # drop '_y' string from each column name of the consistency df, if human_labels == True
  if human_labels == True:
    consistency.columns = consistency.columns.str.replace(r'_y$', '',regex=True)
  # combine the modal label classification to the consistency score
  df_combined = pd.concat([modal_values, consistency], axis=1)
  # reset index
  df_combined = df_combined.reset_index()
  # in the modal label column name, replace '_y' with '_pred'
  if human_labels == True:
    df_combined.columns = [col.replace('_y', '_pred') if '_y' in col else col for col in df_combined.columns]
  # drop first column
  df_combined = df_combined.drop(df_combined.columns[0], axis=1)

  ##### Data cleaning for the input df (combine with )
  # drop duplicates
  df_new = df.drop_duplicates(subset=['unique_id'])
  # replace '_x' with '_pred'
  if human_labels == True:
    df_new.columns = [col.replace('_x', '_true') if '_x' in col else col for col in df_new.columns]
    # add 2 to the length of the categories (to account for unique id and text columns)
    length_of_col = 2 + len(categories_names)
    # clean up columns included in df
    df_new = df_new.iloc[:,0:length_of_col]
    df_new = df_new.reset_index()
    df_new = df_new.drop(df_new.columns[0], axis=1)
  else:
    first_two_columns = df_new.iloc[:, 0:2]
    df_new = pd.DataFrame(first_two_columns)
    df_new = df_new.reset_index()

  # combine into final df
  out = pd.concat([df_new, df_combined], axis=1)


  return out

def evaluate_classification(df, category, num_categories):
  """
  Determines whether the classification is correct, then also adds whether it is a 
  true positive, false positive, true negative, or false negative

  df:
    Input dataframe (text_to_annotate)
  Category:
    Category names to be annotated
  num_categories:
    total number of annotation categories

  Returns:
    Added columns to input dataframe specifying whether the GPT annotation category is correct and whether it is a tp, fp, tn, or fn
  """
  # account for indexing starting at 0
  category = category + 1
  # specify category
  correct = "correct" + "_" + str(category)
  tp = "tp" + "_" + str(category)
  tn = "tn" + "_" + str(category)
  fp = "fp" + "_" + str(category)
  fn = "fn" + "_" + str(category)
  # account for text col and unique id (but already added one to account for zero index start)
  category = category + 1

  # evaluate classification
  df[correct] = (df.iloc[:, category] == df.iloc[:, category+num_categories].astype(int)).astype(int)
  df[tp] = ((df.iloc[:, category] == 1) & (df.iloc[:, category+num_categories].astype(int) == 1)).astype(int)
  df[tn] = ((df.iloc[:, category] == 0) & (df.iloc[:, category+num_categories].astype(int) == 0)).astype(int)
  df[fp] = ((df.iloc[:, category] == 0) & (df.iloc[:, category+num_categories].astype(int) == 1)).astype(int)
  df[fn] = ((df.iloc[:, category] == 1) & (df.iloc[:, category+num_categories].astype(int) == 0)).astype(int)
  
  return df

def performance_metrics(col_names, df):
  """
  Calculates performance metrics (accuracy, precision, recall, and f1) for every category.

  col_names:
    Category names to be annotated
  df:
    Input dataframe (text_to_annotate)

  Returns:
    Dataframe with performance metrics
  """

  # Initialize lists to store the metrics
  categories_names = col_names[1:]
  categories = [index for index, string in enumerate(categories_names)]
  metrics = ['accuracy', 'precision', 'recall', 'f1']
  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_list = []

  # Calculate the metrics for each category and store them in the lists
  for cat in categories:
      tp = df['tp_' + str(cat+1)].sum()
      tn = df['tn_' + str(cat+1)].sum()
      fp = df['fp_' + str(cat+1)].sum()
      fn = df['fn_' + str(cat+1)].sum()
      
      accuracy = (tp + tn) / (tp + tn + fp + fn)
      if tp + fp == 0: # include to account for undefined denominator
        precision = 0
      else:
        precision = tp / (tp + fp)
      if tp + fn == 0: # include to account for undefined denominator
        recall = 0
      else:
        recall = tp / (tp + fn)
      if precision + recall == 0: # include to account for undefined denominator
        f1 = 0 
      else:
        f1 = (2 * precision * recall) / (precision + recall)
      # append metrics
      accuracy_list.append(accuracy)
      precision_list.append(precision)
      recall_list.append(recall)
      f1_list.append(f1)

  # Create a dataframe to store the results
  results = pd.DataFrame({
      'Category': categories_names,
      'Accuracy': accuracy_list,
      'Precision': precision_list,
      'Recall': recall_list,
      'F1': f1_list
  })

  return results

def filter_incorrect(df):
    """
    In order to better understand LLM performance, this function returns a df for all incorrect 
    classifications and all classificatiosn with less than 1.0 consistency.

    df:
      Input dataframe (text_to_annotate)

    Returns:
      Dataframe with incorrect or less than 1.0 consistency scores.
    """

    # Filter rows where any consistency score column value is 1.0
    consistency_cols = [col for col in df.columns if 'consistency' in col]
    consistency_filter = df[consistency_cols] == 1 

    # Filter rows where any correct column value is 1
    correct_cols = [col for col in df.columns if 'correct' in col]
    correct_filter = df[correct_cols] == 1

    # Combine filters
    combined_filter = pd.concat([consistency_filter, correct_filter], axis=1)
    # Filter for any rows where the correct value is not 1 or the consistency is less than 1.0
    mask = combined_filter.apply(lambda x: any(val == False for val in x), axis=1)

    # Apply the filter and return the resulting dataframe
    return df[mask]

def num_tokens_from_string(string: str, encoding_name: str):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def estimate_time_cost(text_to_annotate, codebook, llm_query, 
                       model, num_iterations, num_batches, batch_size, col_names):
  """
  This function estimates the cost and time to run gpt_annotate().

  text_to_annotate:
    Input data that will be annotated.
  codebook:
    String detailing the task-specific instructions.
  llm_query:
    Codebook plus the text samples in batch that will annotated.
  model:
    OpenAI GPT model, which is either gpt-3.5-turbo or gpt-4
  num_iterations:
    Number of iterations in gpt_annotate
  num_batches:
    number of batches in gpt_annotate
  batch_size:
    number of text samples in each batch
  col_names:
    Category names to be annotated

  Returns:
    quit, which is a boolean indicating whether to continue with the annotation process.
  """
  # input estimate
  num_input_tokens = num_tokens_from_string(codebook + llm_query, "cl100k_base")
  total_input_tokens = num_input_tokens * num_iterations * num_batches
  if model == "gpt-4":
    gpt4_prompt_cost = 0.00003
    prompt_cost = gpt4_prompt_cost * total_input_tokens
  else:
    chatgpt_prompt_cost = 0.000002
    prompt_cost = chatgpt_prompt_cost * total_input_tokens

  # output estimate
  num_categories = len(text_to_annotate.columns)-3 # minus 3 to account for unique_id, text, and llm_query
  estimated_output_tokens = 3 + (5 * num_categories) + (3 * batch_size * num_categories) # these estimates are based on token outputs from llm queries
  total_output_tokens = estimated_output_tokens * num_iterations * num_batches
  if model == "gpt-4":
    gpt4_out_cost = 0.00006
    output_cost = gpt4_out_cost * total_output_tokens
  else:
    chatgpt_out_cost = 0.000002
    output_cost = chatgpt_out_cost * total_output_tokens

  cost = prompt_cost + output_cost
  cost_low = round(cost*0.9,2)
  cost_high = round(cost*1.1,2)

  if model == "gpt-4":
    time = round(((total_input_tokens + total_output_tokens) * 0.02)/60, 2)
    time_low = round(time*0.7,2)
    time_high = round(time*1.3,2)
  else:
    time = round(((total_input_tokens + total_output_tokens) * 0.01)/60, 2)
    time_low = round(time*0.7,2)
    time_high = round(time*1.3,2)

  quit = False
  print("You are about to annotate", len(text_to_annotate), "text samples and the number of iterations is set to", num_iterations)
  print("Estimated cost range in US Dollars:", cost_low,"-",cost_high)
  print("Estimated minutes to run gpt_annotate():", time_low,"-",time_high)
  print("Please note that these are rough estimates.")
  print("")
  waiting_response = True
  while waiting_response:
    input_response = input("Would you like to proceed and annotate your data? (Options: Y or N) ")
    input_response = str(input_response).lower()
    if input_response == "y" or input_response == "yes":
      waiting_response = False
    elif input_response == "n" or input_response == "no":
      print("")
      print("Exiting gpt_annotate()")
      quit = True
      waiting_response = False
    else:
      print("Please input Y or N.")
  return quit



