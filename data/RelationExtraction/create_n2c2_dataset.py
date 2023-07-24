import os
from collections import Counter
import re
import copy
import datasets
import pandas as pd
from datasets import concatenate_datasets
from datasets import load_from_disk


def get_paths_lists(data_path, annotations_path):

    data_files = []
    for root, directories, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            data_files.append(file_path)

    annotation_files = []
    for root, directories, files in os.walk(annotations_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            annotation_files.append(file_path)

    return data_files, annotation_files


def get_path_pairs(data_files, annotation_files, file_1=True):

    path_pairs = {}

    for data_path in data_files:
        data_file_number = os.path.basename(data_path)

        for annotation_path in annotation_files:
            annotation_filename = os.path.basename(annotation_path)

            if file_1:
               annotation_file_number = annotation_filename.split(".")[0]
            else:
               annotation_file_number = annotation_filename.split("_")[0]

            if data_file_number == annotation_file_number:
               if data_file_number in path_pairs:
                  print("Duplicate - there is already annotation file for this data file. We continue without assigment.")
                  continue
               else:
                  path_pairs[data_file_number] = [data_path, annotation_path]

    return path_pairs


def read_string(path):

    with open(path, 'rb') as binary_file:
        # Read the contents of the binary file
        binary_data = binary_file.read()

        #  Decode the binary data into a text string
        string = binary_data.decode('utf-8')
    return string

def create_txt_annotation_pairs(path_pairs):

    txt_annotation_pairs = {}

    for key, value in path_pairs.items():
        data_file_path = value[0]
        annotation_file_path = value[1]
        txt_annotation_pairs[key] = [read_string(data_file_path).lower(), read_string(annotation_file_path)]

    return txt_annotation_pairs


def extract_substring_and_numbers(string):
    """
    pattern was desined acording to the n2c2 annotation guidelines
    """
    pattern =  r'\s*(m|do|mo|f|du|r)="([^"]*)" (\d+):(\d+) (\d+):(\d+)'   # example: 'm="aspirin blockade" 35:10 35:11'
    match = re.search(pattern, string)
    if match:
        attribute_type = match.group(1)
        attribute_str = match.group(2)
        positions_numbers = [int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6))]
        return attribute_type, attribute_str, positions_numbers
    else:
        return None

def process_raw_annotations(text_data, annotaion_data):

    attribute_type_mapping = {"m": "Medication",
                              "do": "Dosage",
                              "mo": "Route",
                              "f": "Frequency",
                              "du": "Duration",
                              "r": "Reason"}

    # Extract list of annotations for the given text

    annotations = []

    i = 0
    for raw_annotation in annotaion_data.split("\n"):
        elements = raw_annotation.split("||")
        medication_attributes = []

        for e in elements:
            result = extract_substring_and_numbers(e)
            if result is None:
                continue

            attribute_type, attribute_str, pos_numbers = result
            attribute_str = attribute_str.lower()
            start_row = pos_numbers[0]
            start_word_pos = pos_numbers[1]
            end_row = pos_numbers[2]
            end_word_pos = pos_numbers[3]


            # map attribute position from row:word to text level (word position in text level)
            text_rows = text_data.split("\n")

            # start_word_pos = ' '.join(text_rows[:start_row-1]).count(' ') + start_word_pos + 1
            # end_word_pos = ' '.join(text_rows[:end_row-1]).count(' ') + end_word_pos + 1
            start_word_pos = len(' '.join(text_rows[:start_row-1]).split()) + start_word_pos
            end_word_pos =  len(' '.join(text_rows[:end_row-1]).split()) + end_word_pos

            # Skip examples that are not tagged well
            w1 = text_data.split()[start_word_pos]
            w2 = attribute_str.split()[0]

            w3 = text_data.split()[end_word_pos]
            w4 = attribute_str.split()[-1]

            condition1 = w1 == w2
            condition2 = w3 == w4
            nl = '\n'
            if not condition1:
               print(f"Example is tagged incorectly:{nl} The first word position yields the word: {w1} {nl} while the string annotation is: {w2} {nl} ----------")

               continue

            if not condition2:
               print(f"Example is tagged incorectly:{nl} The first word position yields the word: {w3} {nl} while the string annotation is: {w4} {nl} ----------")
               continue
            #assert text_data.split()[start_word_pos] == attribute_str.split()[0], "start_word_pos is uncorrect"
            #print("|"+text_data.split()[end_word_pos]+"|", "|"+attribute_str.split()[-1]+"|")
            #assert text_data.split()[end_word_pos] == attribute_str.split()[-1], "end_word_pos is uncorrect"


            medication_attributes.append([start_word_pos, end_word_pos, attribute_type_mapping[attribute_type]+str(i), attribute_str])

        # add to annotations only medications that have at least one attribute
        if len(medication_attributes) > 1:
           i += 1
           annotations.append(medication_attributes)

    return annotations


def get_first_char_position(sentence, word_position):
    words = sentence.split()
    char_count = 0

    for i in range(word_position):
        char_count += len(words[i]) + 1  # Add 1 for the space after each word

    start_char_position = char_count

    return start_char_position

def get_last_char_position(sentence, word_position):
    words = sentence.split()
    char_count = 0

    for i in range(word_position):

        char_count += len(words[i]) + 1  # Add 1 for the space after each word

    start_char_position = char_count
    last_char_position = start_char_position + len(words[word_position])

    return last_char_position


def get_context_annotaion_dct(text_data, annotations_lst, context_size):
    # Create dict of {(context_start, context_end): list of attributes in this context}}
    #
    # Example:
    #          {(153, 204): [[156, 156, 'Medication0', 'nitroglycerins'],
    #                       [155, 155, 'Dosage0', 'two'],
    #                       [171, 171, 'Medication1', 'nitroglycerin'],
    #                       [166, 168, 'Reason1', 'escalating chest pain'],
    #                       [181, 181, 'Medication2', 'nitroglycerin.'],
    #                       [175, 175, 'Reason2', 'pain']],
    #
    #            (204, 255): [[219, 219, 'Medication3', 'nitroglycerins'],
    #                         [218, 218, 'Dosage3', 'two']],
    #
    words = text_data.split()
    words_num = len(words)
    context_dct = {}

    for i in range(0, words_num, context_size):
        s = i                                 # context start
        e = min(s + context_size, words_num)  # context end
        for annotaion in annotations_lst:
            for attribute in annotaion:
                atrr_start = attribute[0]
                attr_end = attribute[1]
                if s <= atrr_start < e and s <= attr_end < e:
                  if (s, e) in context_dct:
                      context_dct[(s, e)].append(attribute)
                  else:
                      context_dct[(s, e)] = [attribute]

    # Filter out contexts with only 1 attribute or with no Medication attribute
    new_context_dct = {}
    for c in context_dct:
        if len(context_dct[c]) <= 1:
          continue
        for attribute in context_dct[c]:
            med_in = False
            if "Medication" in attribute[2]:
              new_context_dct[c] = context_dct[c]
    return new_context_dct

def create_examples(text_data, context_dct):
    words = text_data.split()
    examples = []
    for i, context_posistions in enumerate(context_dct):
        start_index = context_posistions[0]
        end_index = context_posistions[1]
        snippet = ' '.join(words[start_index:end_index])

        for annotation in context_dct[context_posistions]:
            # update entities positions to be relative to the snippet and not the whole text
            annotation[0] = annotation[0] - start_index
            annotation[1] = annotation[1] - start_index

            # update entities positions to be start char and end char positions instead of start word and end word positions
            annotation[0] = get_first_char_position(snippet, annotation[0])
            annotation[1] = get_last_char_position(snippet, annotation[1])


        examples.append({"index": i, "snippet": snippet , "annotations": str(context_dct[context_posistions])})

    return examples



def create_n2c2_dataset(txt_annotation_pairs, context_size):
    """

    Parametrs:
       context_size - what would be the max (approx) context size of the snipped. Note that the context would contain at least one Medication.

    """
    examples = []
    for key in txt_annotation_pairs:
        text_data = txt_annotation_pairs[key][0]
        annotaion_data = txt_annotation_pairs[key][1]

        annotations_lst = process_raw_annotations(text_data, annotaion_data)
        context_dct = get_context_annotaion_dct(text_data, annotations_lst, context_size)
        examples += create_examples(text_data, context_dct)

    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=examples))

    return new_dataset

## ------------------------------------------------------------------------------


data_path_1 = "/content/drive/MyDrive/Advanced_NLP_project/n2c2_dataset/test_data/"
annotations_path_1 = "/content/drive/MyDrive/Advanced_NLP_project/n2c2_dataset/converted.noduplicates.sorted/"
# annotations_path_1 = "/content/drive/MyDrive/Advanced_NLP_project/n2c2_dataset/pool/"

data_path_2 = "/content/drive/MyDrive/Advanced_NLP_project/n2c2_dataset/train_data/"
annotations_path_2 = "/content/drive/MyDrive/Advanced_NLP_project/n2c2_dataset/train_gold_truth/"

SNIPPET_SIZE = 51

data_files_1, annotation_files_1 = get_paths_lists(data_path_1, annotations_path_1)
data_files_2, annotation_files_2 = get_paths_lists(data_path_2, annotations_path_2)

path_pairs_1 = get_path_pairs(data_files_1, annotation_files_1, file_1=True)
path_pairs_2 = get_path_pairs(data_files_2, annotation_files_2, file_1=False)

txt_annotation_pairs_1 = create_txt_annotation_pairs(path_pairs_1)
txt_annotation_pairs_2 = create_txt_annotation_pairs(path_pairs_2)

n2c2_dataset_1 = create_n2c2_dataset(txt_annotation_pairs_1, context_size=SNIPPET_SIZE)
n2c2_dataset_2 = create_n2c2_dataset(txt_annotation_pairs_2, context_size=SNIPPET_SIZE)

n2c2_dataset = concatenate_datasets([n2c2_dataset_1, n2c2_dataset_2])

n2c2_dataset.save_to_disk('n2c2_dataset')

#n2c2_dataset = load_from_disk('n2c2_dataset')