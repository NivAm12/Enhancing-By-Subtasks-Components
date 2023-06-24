from datasets import Dataset
import pandas as pd

class AcronymDataset:
    def __init__(self, file_path, tokenizer):
        self._file_path = file_path
        self._dataset = None
        self.tokenizer = tokenizer
        self.preprocessed_dataset = None

        self.__create_dataset()
    
    @property
    def data(self):
        return self._dataset
    
    def __create_dataset(self):
        self.__create_examples()


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
        self._dataset = Dataset.from_dict(data_dict)

    def __create_negative_examples(self):
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
