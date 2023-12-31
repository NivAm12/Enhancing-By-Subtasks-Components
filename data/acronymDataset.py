from datasets import Dataset
import pandas as pd
import random
import pickle
import os
from torch.utils.data import random_split, DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer


class AcronymDataset:
    def __init__(self, file_path: str, tokenizer: AutoTokenizer):
        self._cache_file = "data/acronym_dataset.pkl"
        self._file_path = file_path
        self._dataset = None
        self.tokenizer = tokenizer
        self.preprocessed_dataset = None

        self.__create_dataset()
        self.preprocss_dataset()
    
    @property
    def data(self):
        return self._dataset
                
    def preprocss_dataset(self):
        preprocessed_dataset = self._dataset.map(self.__preprocess_func)
        preprocessed_dataset = preprocessed_dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"])
        self.preprocessed_dataset = preprocessed_dataset

    def get_dataloaders(self, train_size: float=0.9, batch_size: int=32, shuffle: bool=True):
        if self.preprocessed_dataset is None:
            raise ValueError("Preprocessed dataset is not available, create it by using preprocss_dataset before "
                             "using this method.")

        # split the dataset
        self.preprocessed_dataset = self.preprocessed_dataset.class_encode_column(
            "labels")  # cast "label" column to "ClassLabel" type, to support "stratify_by_column" argument below
        split = self.preprocessed_dataset.train_test_split(test_size=1 - train_size, stratify_by_column="labels")
        train_dataset = split["train"]
        val_dataset = split["test"]

        # dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding=True)

        # Create the DataLoaders
        train_dataloader = DataLoader(
                    train_dataset,
                    batch_size = batch_size,
                    collate_fn=data_collator,
                    shuffle=shuffle,
                    drop_last=True,
                )
        
        val_dataloader = DataLoader(
                    val_dataset,
                    batch_size = batch_size,
                    collate_fn=data_collator,
                    shuffle=shuffle,
                    drop_last=True,
                )

        return train_dataloader, val_dataloader

    def __create_dataset(self):
        if os.path.exists(self._cache_file):
            # Load the dataset from cache
            print('[INFO] Dataset already been loaded, using the cached dataset..')
            with open(self._cache_file, "rb") as file:
                self._dataset = pickle.load(file)
        else:
            self.__create_examples()
            self.__create_negative_examples()
            self._dataset = Dataset.from_pandas(self._dataset)
            self._dataset.set_format('torch')

            # Save the dataset to cache for future use
            with open(self._cache_file, "wb") as file:
                pickle.dump(self._dataset, file)

    def __create_examples(self):
        data = []

        with open(self._file_path, "r", errors='ignore') as file:
            for line in file.readlines():
                split = line.strip().split('|')

                # build the sentence structure
                source_sentence = split[6]
                full_name = split[1]
                acronym_begin = int(split[3])
                acronym_end = int(split[4]) + 1

                if len(source_sentence) > 350:
                    continue

                # create the compare sentence with the fit full name
                compare_sentence = source_sentence[:acronym_begin] + full_name + source_sentence[acronym_end:]

                row = {
                    'source_sentence': source_sentence,
                    'compare_sentence': compare_sentence,
                    'labels': 1,
                    'acronym': split[0],
                    'full_name': full_name
                }
                data.append(row)

        data_dict = {key: [item[key] for item in data] for key in data[0]}
        self._dataset = pd.DataFrame.from_dict(data_dict)

    def __create_negative_examples(self):
        groups = self._dataset.groupby('acronym')
        groups_list = []

        # loop each one of the acronym groups
        for _, group in groups:
            # get all of the full names for this acronym group
            full_names = group['full_name'].unique().tolist()
            delete_indexs = []

            # loop over the samples of this group and create a negative sample
            for index, positive_sample in group.iterrows():
                # check if we want to create a negative sample and delete the true sample, because we don't want to keep duplicates of the same sentence
                if random.randint(1, 2) == 1:
                    positive_sample_full_name = positive_sample['full_name']
                    negative_full_names_options = full_names.copy()
                    negative_full_names_options.remove(positive_sample_full_name)

                    if len(negative_full_names_options) > 0:
                        random_false_full_name = random.choice(negative_full_names_options)

                        # create a compare sentence with a false full name
                        true_compare_sentence = positive_sample['compare_sentence']
                        full_name_start_index = true_compare_sentence.find(positive_sample['full_name'])
                        full_name_end_index = full_name_start_index + len(positive_sample['full_name'])

                        false_compare_sentence = true_compare_sentence[:full_name_start_index] + random_false_full_name + true_compare_sentence[full_name_end_index:]
                        negative_example = positive_sample.copy()
                        negative_example['compare_sentence'] = false_compare_sentence
                        negative_example['full_name'] = random_false_full_name
                        negative_example['labels'] = 0
                        
                        # insert it to the group
                        group.loc[len(group)] = negative_example
                        delete_indexs.append(index)

            group = group.drop(delete_indexs)
            groups_list.append(group)  

        # merge the groups again
        self._dataset = pd.concat(groups_list, axis=0)  

    def __preprocess_func(self, example):
        # attach special tokens to the acronym and full name to attract attention from the model
        acronym_start_index = example['source_sentence'].find(example['acronym'])
        acronym_end_index = acronym_start_index + len(example['acronym']) + 1
        full_name_start_index = example['compare_sentence'].find(example['full_name'])
        full_name_end_index = full_name_start_index + len(example['full_name']) + 1

        source_sentence = example['source_sentence'][:acronym_start_index] + '<start>' + example['acronym'] + '<end>' + example['source_sentence'][acronym_end_index:]
        compare_sentence = example['compare_sentence'][:full_name_start_index] + '<start>' + example['full_name']  + '<end>' + example['compare_sentence'][full_name_end_index:]

        result = self.tokenizer(source_sentence, compare_sentence, truncation=True)
        
        return result
