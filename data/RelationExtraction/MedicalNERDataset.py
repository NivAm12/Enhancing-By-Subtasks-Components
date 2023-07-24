
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import numpy as np

import copy
from datasets.arrow_dataset import Dataset


class MedicalNERDataset:

    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer, inference_mode: bool = False):
        self._dataset = dataset
        self.tokenizer = tokenizer
        self.preprocessed_dataset = None
        self._dataloaders = None
        self._inference_mode = inference_mode

        self.id_2_label = {0: "O",
                           1: "B-Medication", 2: "I-Medication",
                           3: "B-Dosage", 4: "I-Dosage",
                           5: "B-Duration", 6: "I-Duration",
                           7: "B-Frequency", 8: "I-Frequency",
                           9: "B-Route", 10: "I-Route",
                           11: "B-Reason", 12: "I-Reason"}

        self.label_2_id = {value: key for key, value in self.id_2_label.items()}

        self.__preprocss_dataset()
        if self._inference_mode:
            self.__create_inference_dataloader()
        else:
            self.__create_dataloaders()

    @property
    def data(self):
        return self._dataset

    def get_dataloaders(self):
        return self._dataloaders

    def __preprocss_dataset(self):
        if self._inference_mode:
            processed_dataset = self._dataset.map(self.__tokenize_and_align_labels,
                                                  fn_kwargs={"tokenizer": self.tokenizer}, batched=False)
        else:
            processed_dataset = self._dataset.map(self.__add_labels)
            processed_dataset = processed_dataset.map(self.__tokenize_and_align_labels,
                                                      fn_kwargs={"tokenizer": self.tokenizer}, batched=False)
        self.preprocessed_dataset = processed_dataset

    def __create_dataloaders(self, train_size: float = 0.9, batch_size: int = 32, shuffle: bool = True):

        if self.preprocessed_dataset is None:
            raise ValueError(
                "Preprocessed dataset is not available, create it by using preprocss_dataset before using this method.")

        # split the dataset
        # train_size = int(train_size * len(self.preprocessed_dataset))
        # val_size = len(self.preprocessed_dataset) - train_size
        # train_dataset, val_dataset = random_split(self.preprocessed_dataset, [train_size, val_size])

        split = self.preprocessed_dataset.train_test_split(test_size=1 - train_size)  # , stratify_by_column="label")
        train_dataset = split["train"]
        val_dataset = split["test"]

        # Dynamic padding (pad each batch seperatley).
        # It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        # label_pad_token_id (int, optional, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="pt",
                                                           padding=True)  # ' label_pad_token_id: int = -100

        col_lst = ["input_ids", "token_type_ids", "attention_mask", "token_type_ids", "labels"]

        train_dataloader = DataLoader(
            train_dataset.select_columns(col_lst),
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0)

        val_dataloader = DataLoader(
            val_dataset.select_columns(col_lst),
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0)

        self._dataloaders = {"train": train_dataloader, "validation": val_dataloader}

    def __create_inference_dataloader(self):

        if self.preprocessed_dataset is None:
            raise ValueError(
                "Preprocessed dataset is not available, create it by using preprocss_dataset before using this method.")

        # Dynamic padding (pad each batch seperatley).
        # It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        # label_pad_token_id (int, optional, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="pt",
                                                           padding=False)  # ' label_pad_token_id: int = -100

        col_lst = ["input_ids", "token_type_ids", "attention_mask", "token_type_ids"]

        inference_dataloader = DataLoader(
            self.preprocessed_dataset.select_columns(col_lst),
            collate_fn=data_collator,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        self._dataloaders = {"inference": inference_dataloader}

    def __create_words_char_position_list(self, string):
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
            end_index = position + word_length  # - 1
            char_indices.append((start_index, end_index))
            position += word_length + 1  # Add 1 to account for the space after the word

        return char_indices

    def __find_tuple_positions_with_numbers(self, tuple_list, s, e):
        """
        Given a list of tuples, find all tuples that contain s or e, and returns the position
        of these tuples in the tuple_list.
        """
        result = []

        for position, tuple_item in enumerate(tuple_list):
            if s == tuple_item[0] or e == tuple_item[1] or e + 1 == tuple_item[1]:
                result.append(position)

        return result

    def __extract_atrribute_name(self, string: str) -> str:
        """
        Given a string in the form {AttributeName}{number} extract the AttributeName.
        AttributeName could be only one of the following: "Medication", "Dosage", "Duration", "Frequency", "Route", "Reason".

        """
        atrr_names = ["Medication", "Dosage", "Duration", "Frequency", "Route", "Reason"]

        for name in atrr_names:
            if name in string:
                return name
        return None

    def __add_labels(self, example):
        """
        Given an example, create a list of BIO labels (each for a token in the sentence) based on the annotation.
        """
        str_sentence = example['snippet']
        sentence_len = len(str_sentence.split(' '))
        labels = np.zeros(sentence_len, dtype='int')

        words_positions = self.__create_words_char_position_list(str_sentence)
        annotations = eval(example['annotations'])

        for annotation in annotations:
            char_start_indx = annotation[0]
            char_end_indx = annotation[1]
            label = self.__extract_atrribute_name(
                annotation[2])  # annotation[2] is the entity type name. Examples: "Medication18", "Dosage0"
            result = self.__find_tuple_positions_with_numbers(words_positions, char_start_indx, char_end_indx)

            if len(result) == 1:  # only one word
                word_idx = result[0]
                labels[word_idx] = self.label_2_id["B-" + label]

            else:  # there are several words
                first_word_idx, last_word_idx = result[0], result[-1]
                labels[first_word_idx:first_word_idx + 1] = self.label_2_id["B-" + label]
                labels[first_word_idx + 1:last_word_idx + 1] = self.label_2_id["I-" + label]

        return {"labels": labels}

    def __tokenize_and_align_labels(self, example, tokenizer):
        """
        Resources:
        https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task
        https://huggingface.co/docs/transformers/tasks/token_classification
        https://www.lighttag.io/blog/sequence-labeling-with-transformers/example

        Tokinize the tokens and then realign the tokens and labels by:

        1. Mapping all tokens to their corresponding word with the word_ids method.
        2. Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function.
        3. Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.

        For example if the word "runing" is tokinized to two words "run" and "#ing", we assign the label only to the first word ("run"), and assign "-100" to the other word ("#ing).

        Returns:
            tokenized_inputs - a dict that represents the tokinized input, including the new label list:
                                ({'input_ids', 'token_type_ids', 'attention_mask', 'labels})

        """
        tokenized_inputs = tokenizer(example["snippet"].split(), is_split_into_words=True, truncation=True,
                                     max_length=512)  # , padding=True)#, return_tensors="pt")

        # Map tokens to their respective word.
        # word_ids is a list indicating the word corresponding to each token. Special tokens added by
        # the tokenizer are mapped to None and other tokens are mapped to the index of their
        # corresponding word (several tokens will be mapped to the same word index if they are parts of that word).
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        crf_mask = copy.deepcopy(tokenized_inputs["attention_mask"])

        for i, word_idx in enumerate(word_ids):
            # Set the special tokens ([CLS] and [SEP]) to crf_mask=0
            if word_idx is None:
                crf_mask[i] = 0
            #  Label the first token of a given word with mask==1, and assign 0 to other subtokens
            elif word_idx != previous_word_idx:
                pass
            else:
                crf_mask[i] = 0
            previous_word_idx = word_idx

        tokenized_inputs["token_type_ids"] = crf_mask

        # align labels if not inference_model
        if not self._inference_mode:

            old_labels_list = example["labels"]
            new_labels_list = []
            previous_word_idx = None

            for i, word_idx in enumerate(word_ids):
                # Set the special tokens ([CLS] and [SEP]) to -100.
                if word_idx is None:
                    new_labels_list.append(-100)
                #  Label the first token of a given word, and assign -100 to other subtokens
                elif word_idx != previous_word_idx:
                    new_labels_list.append(old_labels_list[word_idx])
                else:
                    new_labels_list.append(-100)  # -100   #old_labels_list[word_idx]
                previous_word_idx = word_idx

            tokenized_inputs["labels"] = new_labels_list

        return tokenized_inputs

    def print_tokenized_words_and_labels(self, example):
        """

        """
        tokinized_words = ' '.join([self.tokenizer.decode(input_id) for input_id in example['input_ids']]).split(' ')
        print("token    |   label     |  attention_mask   | crf_mask")
        print("______________________________________________________")
        for i, word in enumerate(tokinized_words):
            if example['labels'][i] == -100:
                print(word, "      |", -100, "|", example['attention_mask'][i], "|", example['token_type_ids'][i])
            else:
                print(word, "      |", self.id_2_label[example['labels'][i]], "|", example['attention_mask'][i], "|",
                      example['token_type_ids'][i])


