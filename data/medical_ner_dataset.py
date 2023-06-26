
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader

from TorchCRF import CRF
import torch
import torch.nn as nn
import numpy as np

id_2_label = {0 : "O",
             1 : "B-Medication", 2 : "I-Medication",
             3 : "B-Dosage",     4 : "I-Dosage",
             5 : "B-Duration",   6 : "I-Duration",
             7 : "B-Frequency",  8 : "I-Frequency",
             9 : "B-Route",      10 : "I-Route",
             11 : "B-Reason",    12 : "I-Reason"}

label_2_id = switched_dict = {value: key for key, value in id_2_label.items()}


def create_words_char_position_list(string):
    """
    Given a string of words, returns a list of tuples as the numbers of words.
    Each tuple contains the start and end char index of the word in the given string.
    Example: string = "Go home" ---> [(0,1), (3,6)]
    """
    words = string.split()
    char_indices = []
    position = 0

    for word in words:
        word_length = len(word)
        start_index = position
        end_index = position + word_length - 1
        char_indices.append((start_index, end_index))
        position += word_length + 1  # Add 1 to account for the space after the word

    return char_indices

def find_tuple_positions_with_numbers(tuple_list, s, e):
    """
    Given a list of tuples, find all tuples that contain s or e, and returns the position
    of these tuples in the tuple_list.
    """
    result = []

    for position, tuple_item in enumerate(tuple_list):
        if s in tuple_item or e in tuple_item:
            result.append(position)

    return result

def add_labels(example):
    """
    Given an example, create a list of BIO labels (each for a token in the sentence) based on the annotation.
    """
    str_sentence = example['snippet']
    sentence_len = len(str_sentence.split(' '))
    labels = np.zeros(sentence_len, dtype='int')

    words_positions = create_words_char_position_list(str_sentence)

    annotations = eval(example['annotations'])

    for annotation in annotations:
        char_start_indx = annotation[0]
        char_end_indx = annotation[1]
        label = annotation[2][:-1]

        result = find_tuple_positions_with_numbers(words_positions, char_start_indx, char_end_indx)

        if len(result) == 1: # only one word
           word_idx = result[0]
           labels[word_idx] = label_2_id["B-"+label]

        else: # there are several words
          first_word_idx, last_word_idx = result
          labels[first_word_idx:first_word_idx+1] = label_2_id["B-"+label]
          labels[first_word_idx+1:last_word_idx+1] = label_2_id["I-"+label]

    return {"labels" : labels}

def tokenize_and_align_labels(example, tokenizer):
    """
    Sources:
    https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task
    https://huggingface.co/docs/transformers/tasks/token_classification

    Tokinize the tokens and then realign the tokens and labels by:

    1. Mapping all tokens to their corresponding word with the word_ids method.
    2. Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function.
    3. Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.

    For example if the word "runing" is tokinized to two words "run" and "#ing", we assign the label only to the first word ("run"), and assign "-100" to the other word ("#ing).

    @Returns: dict that represents the tokinized input ({'input_ids', 'token_type_ids', 'attention_mask'}) + updated labels

    """
    tokenized_inputs = tokenizer(example["snippet"], truncation=True, padding=True)#, return_tensors="pt")
    #print(tokenized_inputs)
    labels = []


    for i, label in enumerate(example["labels"]):
        #print(i, label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        #print(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(word_idx) #(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == '__main__':
    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset_atrr = load_dataset("mitclinicalml/clinical-ie", "medication_attr")
    dataset_atrr_train_raw = dataset_atrr["validation"]
    dataset_atrr_test_raw = dataset_atrr["test"]

    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_atrr_train = dataset_atrr_train_raw.map(add_labels)

    tokinized_dataset_atrr_train = dataset_atrr_train.map(tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer},
                                                          batched=True)
    # Dynamic padding (pad each batch seperatley).
    # It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
    # label_pad_token_id (int, optional, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt", padding=True)

    trainLoader = DataLoader(
        tokinized_dataset_atrr_train.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"]),
        collate_fn=data_collator,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=0)
