from transformers import AutoConfig, EvalPrediction, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
from evaluate import load
import numpy as np
# import wandb
import os
import sys
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=2)

file_path = 'data/acronym_data.txt'
data = []

with open(file_path, "r", errors='ignore') as file:
    for line in file.readlines():
        split = line.strip().split('|')
        
        # build the sentence structure
        source_sentence = split[6]
        compare_sentence = source_sentence[:int(split[3])] + split[1] + source_sentence[int(split[4]):]

        row = {
            'source_sentence': source_sentence,
            'compare_sentence': compare_sentence,
            'label': 1
        }
        data.append(row)

        # this is how we classify 
        # tokens = tokenizer(source_sentence, compare_sentence, return_tensors='pt')
        # output = model(**tokens)

# Transform the list of dictionaries into a dictionary of lists
data_dict = {key: [item[key] for item in data] for key in data[0]}

# Create a new dataset from the dictionary
dataset = Dataset.from_dict(data_dict)

print('aa')