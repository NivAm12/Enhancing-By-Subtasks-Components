from tqdm import tqdm
from datasets import load_dataset
import torch
from data.RelationExtraction.MedicalNERDataset import MedicalNERDataset


# ---------------------------------------------------------------------------------------------

def add_e1_e2_tokens(example):
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

def tokinize_and_add_e1_e2_positions(example, tokenizer, pubmedbert):

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


# ---------------------------------------------------------------------------------------------

def process_preds_names(preds):
    res = []
    for p in preds:
        if "Medication" in p:
           res.append("Medication")
        elif "Dosage" in p:
           res.append("Dosage")
        elif "Duration" in p:
           res.append("Duration")
        elif "Frequency" in p:
           res.append("Frequency")
        elif "Route" in p:
           res.append("Route")
        elif "Reason" in p:
           res.append("Reason")
        else: # "O"
           res.append("O")
    return res


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

def find_consecutive_occurrences(input_list):
    result = []
    start = 0
    for i in range(1, len(input_list)):
        if input_list[i] != input_list[i - 1]:
            result.append([start, i - 1, input_list[i]])
            start = i
    result.append([start, len(input_list) - 1, input_list[i]])
    return result

def get_annotaions(lst, snippet):
    result = []
    for item in lst:
        start_pos = get_first_char_position(snippet, item[0])
        end_pos = get_last_char_position(snippet, item[1])

        result.append([start_pos, end_pos, item[2], snippet[start_pos:end_pos]])
    return result


def process_t_annotations(annotations):
    processed_true_annotations = []
    prev_num = -1
    i = -1
    for entity in eval(annotations):
        num = entity[2][-1]
        entity[2] = entity[2][0:-1]
        if num == prev_num:
           processed_true_annotations[i].append(entity)
        else:
            i += 1
            prev_num = num
            processed_true_annotations.append([entity])

    return processed_true_annotations


def entity_type_exists(lst, e_type):
    for entity in lst:
        if e_type in entity[2]:
          return True
    return False

def process_t_annotations(annotations):
    processed_true_annotations = []
    prev_num = -1
    i = -1
    for entity in eval(annotations):
        num = entity[2][-1]
        entity[2] = entity[2][0:-1]
        entity_type = entity[2]
        if num == prev_num:
           if not entity_type_exists(processed_true_annotations[i], entity_type): # if entity with the same type already exist, don't add another
              processed_true_annotations[i].append(entity)
        else:
            i += 1
            prev_num = num
            processed_true_annotations.append([entity])

    return processed_true_annotations


# ------------------------------------  Score functions --------------------------------------------------

MED = 0
MED_START = 0
MED_END = 1
ATTRIBUTE_TYPE = 2
ATTRIBUTE_START = 0
ATTRIBUTE_END = 1


def sort_annotation(annotation):
    """
    :param annotation: true or predicted annotation
     sort each list such that the Medication attribute will come first
    """
    for med in annotation:
        for i in range(len(med)):
            if med[i][ATTRIBUTE_TYPE] == "Medication":
                temp = med[i]
                med[i] = med[MED]
                med[MED] = temp
                break


def find_most_fit_true_med(pred_med, true_annotation):
    most_fit, most_fit_score = None, 0
    for true_med in true_annotation:
        # check if the predicted med and the true med have overlap spans
        if (pred_med[MED][MED_START] <= true_med[MED][MED_START] <= pred_med[MED][MED_END]) or (
                true_med[MED][MED_START] <= pred_med[MED][MED_START] <= true_med[MED][MED_END]):
            # we want to relate every predicted med the true me with the largest overlap span
            fit_score = 1 + min(pred_med[MED][MED_END], true_med[MED][MED_END]) - \
                        max(pred_med[MED][MED_START], true_med[MED][MED_START])
            if fit_score > most_fit_score:
                most_fit_score = fit_score
                most_fit = true_med
    return most_fit


def find_most_fit_attribute(pred_att, true_med):
    most_fit, most_fit_score = None, 0
    for att in true_med:
        # both attributes have the same type
        if pred_att[ATTRIBUTE_TYPE] == att[ATTRIBUTE_TYPE]:
            # check if both attributes have also overlap spans
            if (pred_att[ATTRIBUTE_START] <= att[ATTRIBUTE_START] <= pred_att[ATTRIBUTE_END]) or (
                    att[ATTRIBUTE_START] <= pred_att[ATTRIBUTE_START] <= att[ATTRIBUTE_END]):
                # we will choose the true attribute with the largest overlap span for the predicted one
                fit_score = 1 + min(pred_att[ATTRIBUTE_END], att[ATTRIBUTE_END]) - \
                            max(pred_att[ATTRIBUTE_START], att[ATTRIBUTE_START])
                if fit_score > most_fit_score:
                    most_fit_score = fit_score
                    most_fit = att
    return most_fit


def calculate_tp_fp_per_med(pred_med, true_med):
    tp, fp = 0., 0.
    if true_med is not None:
        for pred_att in pred_med:
            true_att = find_most_fit_attribute(pred_att, true_med)
            if true_att is not None:
                # each span as the weight of 1 we will take the number of truly predicted tokens by the number of
                # predicted tokens as true positive
                att_tp = (1 + float(min(pred_att[ATTRIBUTE_END], true_att[ATTRIBUTE_END]) -
                                    max(pred_att[ATTRIBUTE_START], true_att[ATTRIBUTE_START]))) \
                         / (1 + pred_att[ATTRIBUTE_END] - pred_att[ATTRIBUTE_START])
                tp += att_tp
                fp += 1 - att_tp
            else:
                fp += 1
    else:
        # the medication was predicted wrong so all its attributes are wrong
        fp += len(pred_med)
    return tp, fp


def calculate_tp_fp(predicted_annotation, true_annotation):
    tp, fp = 0., 0.
    for pred_med in predicted_annotation:
        true_med = find_most_fit_true_med(pred_med, true_annotation)
        med_tp, med_fp = calculate_tp_fp_per_med(pred_med, true_med)
        tp += med_tp
        fp += med_fp
    return tp, fp


def score(predicted_annotation, true_annotation):
    sort_annotation(predicted_annotation)
    sort_annotation(true_annotation)
    tp, fp = calculate_tp_fp(predicted_annotation, true_annotation)
    # if we look at the true annotation as the predicted ones and opposite then the false positive of this case is
    # the false negative of the original case
    _, fn = calculate_tp_fp(true_annotation, predicted_annotation)
    # F1 score formula
    return 2 * tp / (2 * tp + fp + fn)



# -------------------------------------------------------------------------------------


def inference(crf_model, rc_model, ner_inference_dataloader, tokenizer, pubmedbert, device):

    print("Device:", device)
    crf_model.to(device).eval()
    rc_model.to(device).eval()

    final_result = []

    for i, example in enumerate(tqdm(ner_inference_dataloader)):
      example_metadata = inference_metadata[i]

      # ---- NER Model ----------
      preds = crf_model(input_ids=example["input_ids"].to(device), attention_mask=example["attention_mask"].to(device), crf_mask=example["token_type_ids"].to(device))
      snippet = example_metadata["snippet"]
      preds_names = [medical_ner_dataset_inference.id_2_label[p] for p in preds]
      ppreds = process_preds_names(preds_names)
      words_positions = find_consecutive_occurrences(ppreds)
      annotations = get_annotaions(words_positions, snippet)

      medication_lst = [annotation for annotation in annotations if "Medication" in annotation[2]]
      atributes_lst = [annotation for annotation in annotations if "Medication" not in annotation[2]]

      # --------RC Model ------------
      result = []
      for j in range(len(medication_lst)):
          annotation_lst = [medication_lst[j]]
          for k in range(len(atributes_lst)):
              if 'O' != atributes_lst[k][2]:
                rc_example = {"snippet": snippet, "entity1": str(medication_lst[j]), "entity2" : str(atributes_lst[k])}
                processed_rc_example = add_e1_e2_tokens(rc_example)
                tokinized_rc_example = tokinize_and_add_e1_e2_positions(processed_rc_example, tokenizer, pubmedbert)

                # pred is "0" or "1". Indicator to wether the given medication and attriubute are related or not.
                pred = rc_model.predict(torch.tensor(tokinized_rc_example["input_ids"]).unsqueeze(0).to(device),
                                        torch.tensor(tokinized_rc_example["attention_mask"]).unsqueeze(0).to(device),
                                        torch.tensor(tokinized_rc_example["e1_start_pos"]).unsqueeze(0).to(device),
                                        torch.tensor(tokinized_rc_example["e2_start_pos"]).unsqueeze(0).to(device))

                if pred:
                  annotation_lst.append(atributes_lst[k])

          result.append(annotation_lst)

      final_result.append({"index": i, "snippet": snippet, "true_annotations" : process_t_annotations(example_metadata["annotations"]) , "pred_annotations": result})

    return final_result

def calculate_score(results):

    total_score = 0.0
    examples_num = len(results)

    for i in range(examples_num):
        example = results[i]
        true_annotations = example["true_annotations"]
        pred_annotations = example["pred_annotations"]
        example_score = score(pred_annotations, true_annotations)
        print(example_score)
        total_score += example_score


    return total_score / examples_num



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO
    crf_model =
    rc_model =
    tokenizer =
    pubmedbert =


    # Load Data from Hugginface
    # get token from: https://huggingface.co/settings/tokens (need to be loggen in to my hugginface account)
    from huggingface_hub import login
    login()

    dataset_atrr = load_dataset("mitclinicalml/clinical-ie", "medication_attr")
    clinicallm_dataset_test = dataset_atrr["test"]

    # Process test dataset
    medical_ner_dataset_inference = MedicalNERDataset(clinicallm_dataset_test, tokenizer, inference_mode=True)
    ner_inference_dataloader = medical_ner_dataset_inference.get_dataloaders()["inference"]
    inference_metadata = medical_ner_dataset_inference.preprocessed_dataset

    # Do inference and calculate score
    results = inference(crf_model, rc_model, ner_inference_dataloader, tokenizer, pubmedbert, device)
    score = calculate_score(results)
    print("Test score:", score)










