from transformers import AutoConfig, EvalPrediction, AutoTokenizer, AutoModelForMaskedLM, Trainer, \
    TrainingArguments, set_seed
from datasets import load_dataset, Dataset
from evaluate import load
import numpy as np
# import wandb
import os
import sys


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

file_path = 'data/acronym_data.txt'
data = []

with open(file_path, "r", errors='ignore') as file:
    for line in file.readlines():
        split = line.strip().split('|')
        row = {
            'acronym': split[0],
            'full_name': split[1],
            'location': (int(split[3]), int(split[4])),
            'text': split[6]
        }
        data.append(row)

# Transform the list of dictionaries into a dictionary of lists
data_dict = {key: [item[key] for item in data] for key in data[0]}

# Create a new dataset from the dictionary
dataset = Dataset.from_dict(data_dict)

print('aa')