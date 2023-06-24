from datasets import Dataset
import pandas as pd
import random
import pickle
import os


class AcronymDataset:
    def __init__(self, file_path, tokenizer):
        self._cache_file = "data/acronym_dataset.pkl"
        self._file_path = file_path
        self._dataset = None
        self.tokenizer = tokenizer
        self.preprocessed_dataset = None

        self.__create_dataset()
    
    @property
    def data(self):
        return self._dataset
    
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

                # create the compare sentence with the fit full name
                compare_sentence = source_sentence[:acronym_begin] + full_name + source_sentence[acronym_end:]

                row = {
                    'source_sentence': source_sentence,
                    'compare_sentence': compare_sentence,
                    'label': 1,
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

            # loop over the samples of this group and create a negative sample
            for _, positive_sample in group.iterrows():
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
                    negative_example['label'] = 0
                    
                    # insert it to the group
                    group.loc[len(group)] = negative_example

            groups_list.append(group)  

        # merge the groups again
        self._dataset = pd.concat(groups_list, axis=0)

    def __remove_duplicates(self):
        pass                

    def preprocss_dataset(self):
        preprocessed_dataset = self._dataset.map(self.__preprocess_func) 
        preprocessed_dataset.set_format('torch') 
        preprocessed_dataset = preprocessed_dataset.remove_columns(['source_sentence', 'compare_sentence', 'acronym', 'full_name'])

        self.preprocessed_dataset = preprocessed_dataset

    def __preprocess_func(self, examples):
        # attach special tokens to the acronym and full name to attract attention from the model
        acronym_start_index = examples['source_sentence'].find(examples['acronym'])
        acronym_end_index = acronym_start_index + len(examples['acronym']) + 1
        full_name_start_index = examples['compare_sentence'].find(examples['full_name'])
        full_name_end_index = full_name_start_index + len(examples['full_name']) + 1

        source_sentence = examples['source_sentence'][:acronym_start_index] + '<start>' + examples['acronym'] + '<end>' + examples['source_sentence'][acronym_end_index:]
        compare_sentence = examples['compare_sentence'][:full_name_start_index] + '<start>' + examples['full_name']  + '<end>' + examples['compare_sentence'][full_name_end_index:]

        result = self.tokenizer(source_sentence, compare_sentence, truncation=True, return_tensors='pt')
        
        return result
