import datasets
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import pandas as pd
import random
from datasets.arrow_dataset import Dataset

class MedicalRCDataset:

    def __init__(self, dataset : Dataset, tokenizer: AutoTokenizer, pubmedbert, batch_size: int = 32, train_size: float = 0.9):
        self._dataset = dataset
        self.tokenizer = tokenizer
        self.pubmedbert = pubmedbert
        self.preprocessed_dataset = None
        self._dataloaders = None

        self.__preprocss_dataset()
        self.__create_dataloaders(train_size=train_size, batch_size=batch_size)


    @property
    def data(self):
        return self._dataset

    def get_dataloaders(self):
        return self._dataloaders

    def __preprocss_dataset(self):

        processed_dataset = self.__create_RC_dataset(self._dataset, balance_pos_neg=False)
        processed_dataset = processed_dataset.map(self.__add_e1_e2_tokens)
        processed_dataset = processed_dataset.map(self.__tokinize_and_add_e1_e2_positions, fn_kwargs={"tokenizer": self.tokenizer, "pubmedbert":self.pubmedbert}, batched=False)
        self.preprocessed_dataset = processed_dataset

    def __create_dataloaders(self, train_size: float=0.9, batch_size: int=32, shuffle: bool=True):

        if self.preprocessed_dataset is None:
            raise ValueError("Preprocessed dataset is not available, create it by using preprocss_dataset before using this method.")


        # split the dataset
        #train_size = int(train_size * len(self.preprocessed_dataset))
        #val_size = len(self.preprocessed_dataset) - train_size
        #train_dataset, val_dataset = random_split(self.preprocessed_dataset, [train_size, val_size])

        self.preprocessed_dataset = self.preprocessed_dataset.class_encode_column("label") # cast "label" column to "ClassLabel" type, to support "stratify_by_column" argument below
        split = self.preprocessed_dataset.train_test_split(test_size=1-train_size, stratify_by_column="label")
        train_dataset = split["train"]
        val_dataset = split["test"]

        # Dynamic padding (pad each batch seperatley).
        # It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        # label_pad_token_id (int, optional, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding=True)

        train_dataloader = DataLoader(
                                train_dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", 'e1_start_pos', 'e2_start_pos', "label"]),
                                collate_fn=data_collator,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=False,
                                num_workers=0)

        val_dataloader = DataLoader(
                              val_dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "label"]),
                              collate_fn=data_collator,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=False,
                              num_workers=0)



        self._dataloaders = {"train":train_dataloader, "validation":val_dataloader}

    # ---------------- Create RC dataset functions ----------------------------------

    def __create_medication_attributes_dct(self, annotations):
        """
        Create a dictionary that contains as key the medication index , and as values a list of attributes annotations
        (where the first attribute is actually the medication itself)

        Parametrs:
          annotations: list of list. Example:

                        [[3, 11, 'Medication0', 'Cellcept'],
                          [12, 18, 'Dosage0', '500 mg'],
                          [19, 23, 'Route0', 'p.o.'],
                          [24, 30, 'Frequency0', 'q.i.d.'],
                          [35, 43, 'Medication1', 'Nystatin'],
                          [44, 49, 'Dosage1', '10 ml'],
                          [50, 54, 'Route1', 'p.o.'],
                          [55, 61, 'Frequency1', 'q.i.d.']]

        Returns:
            medication_attributes_dct: dictionary. Example:

                        {'0': [[3, 11, 'Medication0', 'Cellcept'],
                                [12, 18, 'Dosage0', '500 mg'],
                                [19, 23, 'Route0', 'p.o.'],
                                [24, 30, 'Frequency0', 'q.i.d.']],

                          '1': [[35, 43, 'Medication1', 'Nystatin'],
                                [44, 49, 'Dosage1', '10 ml'],
                                [50, 54, 'Route1', 'p.o.'],
                                [55, 61, 'Frequency1', 'q.i.d.']]}

        """

        medication_attributes_dct = {}
        for annotation in annotations:
            index = annotation[2][-1]

            if index in medication_attributes_dct:
              medication_attributes_dct[index].append(annotation)
            else:
              medication_attributes_dct[index] = [annotation]

        return medication_attributes_dct


    def __create_pos_neg_examples(self, snippet, medication_attributes_dct):
        """
        Extract from a single example a list of positive and negative examples, in the required format.

        Parametrs:
            snippet: the snippet text of the example
            medication_attributes_dct: dictionary of medications and attributes.


            Example:

                        {'0': [[3, 11, 'Medication0', 'Cellcept'],
                                [12, 18, 'Dosage0', '500 mg'],
                                [19, 23, 'Route0', 'p.o.'],
                                [24, 30, 'Frequency0', 'q.i.d.']],

                          '1': [[35, 43, 'Medication1', 'Nystatin'],
                                [44, 49, 'Dosage1', '10 ml'],
                                [50, 54, 'Route1', 'p.o.'],
                                [55, 61, 'Frequency1', 'q.i.d.']]}


        Returns:
            examples: Two lists of positive and negative examples. Example:

              positive_examples =   [{'snippet': '...',
                                      'entity1': '[3, 11, 'Medication0', 'Cellcept']',
                                      'entity2': '[12, 18, 'Dosage0', '500 mg']',
                                      'label': 1}, ]

              negative_examples =   [{'snippet': '...',
                                      'entity1': '[35, 43, 'Medication1', 'Nystatin']',
                                      'entity2': '['211, 217, 'Frequency0', 'q.i.d.']',
                                      'label': 0}, ]

        """

        positive_examples = []
        negative_examples = []
        # For each index, create positive and negative examples
        for i in medication_attributes_dct:

            i_medication_annotation = medication_attributes_dct[i][0] # medication is always the first entity in medication_attributes_dct
            i_attributes_annotations = medication_attributes_dct[i][1:]

            # Create positive examples for index i : format:      {'snippet': snippet,
            #                                                      'entity1': '[position, Medication_i, words_in_text]'
            #                                                      'entity2': '[position, Attribute_i, words_in_text]'
            #                                                      'label': 1}

            for i_attribute_annotation in i_attributes_annotations:
                positive_example = {}
                positive_example['snippet'] =  snippet
                positive_example['entity1'] = str(i_medication_annotation)
                positive_example['entity2'] = str(i_attribute_annotation)
                positive_example['label'] = 1
                positive_examples.append(positive_example)

            # Create negative examples for index i : format:      {'snippet': snippet,
            #                                                      'entity1': '[position, Medication_i, words_in_text]'
            #                                                      'entity2': '[position, Attribute_j, words_in_text]'
            #                                                      'label': 0}

            for j in medication_attributes_dct:
                if j == i:
                    continue

                j_attributes_annotations = medication_attributes_dct[j][1:]

                for j_attribute_annotation in j_attributes_annotations:
                    negative_example = {}
                    negative_example['snippet'] =  snippet
                    negative_example['entity1'] = str(i_medication_annotation)
                    negative_example['entity2'] = str(j_attribute_annotation)
                    negative_example['label'] = 0
                    negative_examples.append(negative_example)

        return positive_examples, negative_examples


    def __create_RC_dataset(self, dataset, balance_pos_neg=True):
        """
        Parametrs:
          dataset
          balance_pos_neg: indicate whatever to balance the positive and negative examples

          Returns:
              new_dataset - Dataset with columns: snippet, entity1, entity2, label

        """
        positive_examples_lst = []
        negative_examples_lst = []
        for example in dataset:
            snippet = example['snippet']
            annotations = eval(example['annotations'])

            medication_attributes_dct = self.__create_medication_attributes_dct(annotations)
            positive_examples, negative_examples = self.__create_pos_neg_examples(snippet, medication_attributes_dct)
            positive_examples_lst += positive_examples
            negative_examples_lst += negative_examples

        if balance_pos_neg:
            n = len(positive_examples_lst)
            negative_examples_lst = random.sample(negative_examples_lst, n) # choose n elements from list

        examples = positive_examples_lst + negative_examples_lst
        new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=examples))

        return new_dataset

      # ----------------------- preprocessing functions ---------------------


    def __add_e1_e2_tokens(self, example):
        """
        Given an example, insert e1_start, e1_end, e2_start and e2_end tokens around two entities
        """
        str_sentence = example['snippet']

        entity1 = eval(example['entity1'])
        entity2 = eval(example['entity2'])

        if entity2[0] < entity1[0]: # define whos the first string
            entity2, entity1 = entity1, entity2

        start1_pos = entity1[0]
        end1_pos = entity1[1]
        start2_pos = entity2[0]
        end2_pos = entity2[1]

        start1_symbol = "[E1]"
        end1_symbol =  "[/E1]"
        start2_symbol = "[E2]"
        end2_symbol =  "[/E2]"

        # Add start symbol before the substring
        updated_string = str_sentence[:start1_pos]  + start1_symbol + " " + str_sentence[start1_pos:]

        # Adjust the start and end position based on the added characters
        start1_pos += len(start1_symbol)
        end1_pos += len(start1_symbol) + 1

        # Add end symbol after the substring
        updated_string = updated_string[:end1_pos] + " " + end1_symbol  + updated_string[end1_pos:]


        start2_pos += len(start1_symbol) + len(end1_symbol) + 2
        end2_pos += len(start1_symbol) + len(end1_symbol) + 2

        # Add start symbol before the substring
        updated_string = updated_string[:start2_pos]  + start2_symbol + " " + updated_string[start2_pos:]
        # Adjust the start and end position based on the added characters
        start2_pos += len(start2_symbol)
        end2_pos += len(start2_symbol) + 1
        # Add end symbol after the substring
        updated_string = updated_string[:end2_pos] + " " + end2_symbol  + updated_string[end2_pos:]

        updated_string = updated_string.strip()
        example['snippet'] = updated_string

        # remove entities positions cuz they are not updated anymore (alternatlivy we could update them but we dont use them anyway..)
        example["entity1"] = str(eval(example["entity1"])[2:])
        example["entity2"] = str(eval(example["entity2"])[2:])

        return example

    def __tokinize_and_add_e1_e2_positions(self, example, tokenizer, pubmedbert):

        start1_symbol = "[E1]"
        end1_symbol =  "[/E1]"
        start2_symbol = "[E2]"
        end2_symbol =  "[/E2]"

        num_added_toks = tokenizer.add_tokens([start1_symbol, end1_symbol, start2_symbol, end2_symbol])
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        new_embedding_size = pubmedbert.resize_token_embeddings(len(tokenizer))

        tokenized_inputs = tokenizer(example["snippet"].split(), is_split_into_words=True, truncation=True, max_length=512)#, padding=True)#, return_tensors="pt")

        e1_start_id = tokenizer.convert_tokens_to_ids(start1_symbol)
        e2_start_id = tokenizer.convert_tokens_to_ids(start2_symbol)

        tokenized_inputs["e1_start_pos"] = tokenized_inputs["input_ids"].index(e1_start_id)
        tokenized_inputs["e2_start_pos"] = tokenized_inputs["input_ids"].index(e2_start_id)

        return tokenized_inputs
