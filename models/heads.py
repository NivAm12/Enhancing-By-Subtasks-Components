import torch.nn as nn
import torch
from TorchCRF import CRF
import numpy as np


class ClassificationHead(nn.Module):
    """
    A classification head for a language model.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.25, inplace=False)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = None if out_features == 1 else nn.Softmax()

    def forward(self, inputs, batch=None):
        outputs = self.dropout(inputs.pooler_output)
        outputs = self.linear(inputs.pooler_output)  # pooler_output is the index of cls token
        outputs = outputs if self.activation is None else self.activation(outputs)

        return outputs


# --------------------------------------------------------------------------------------------------------


class NERHead(nn.Module):
    # CRF explained:
    # https://hyperscience.com/blog/exploring-conditional-random-fields-for-nlp-applications/
    # https://createmomo.github.io/archives/2017/10/

    def __init__(self, hidden_size, num_labels):
        super(NERHead, self).__init__()
        self.hidden_size = hidden_size  # last hidden state of BERT model
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.activation = nn.ReLU()
        self.crf = CRF(num_labels)  # , batch_first = True)

    def forward(self, inputs, batch=None):
        """
        input_ids: shape - (batch_size, seq_len)
        labels - if note None, calculate loss. Else, calculate predictions.
        """
        crf_mask = batch['token_type_ids']
        embeddings = inputs["last_hidden_state"]
        embeddings = self.dropout(embeddings)
        emission_scores = self.classifier(embeddings)  # shape: (batch_size, seq_len, num_labels)
        emission_scores = self.activation(emission_scores)
        labels = batch['labels'] if batch is not None else None

        if labels is not None:
            # we put labels with -100 to 0 because crf handles only labels in range(0, num_labels). Since this label
            # is incorret, we also create crf_mask that have 0 in indexes where lable==-100. Like that, the crf will
            # ignore labels with -100 in the loss calculaion. Note that we dont change the mask before the BERT
            # layer, because we want to get information also from tokens with -100.
            crf_mask_modified = crf_mask.clone()
            crf_mask_modified[labels == -100] = 0
            labels_modified = labels.clone()
            labels_modified[labels == -100] = 0

            # loss for each example in the batch. shape: [batch_size] (tensor of length batch_size).
            # the crf loss for each example is calculated considering the true against the predicted sequence of labels,
            loss = self.crf.forward(emission_scores, labels_modified,
                                    crf_mask_modified.type('torch.BoolTensor').to(labels.device))

            return -loss.mean()

        else:
            # Return predictions for each example. shape: [batch_size, seq_len]. NOTE: for each example in batch,
            # it actually not return labels of length seq_len, but only return prediction for tokens that have
            # crf_mask==1, so we process the raw_predictions by taking predictions with crf_mask==1 (crf_mask==0 for
            # spesicel tokens (CLS,SEP), padding tokens, and subtokens that splitted by the tokinizer (has '##'
            # symbol before them))
            raw_predictions = self.crf.viterbi_decode(emission_scores, mask=crf_mask.type('torch.BoolTensor'))
            predictions = self.__process_predictions(raw_predictions, crf_mask)

            return predictions

    def __process_predictions(self, preds, mask):
        flatten_mask = mask.flatten().cpu().numpy()
        flatten_preds = [item for sublist in preds for item in sublist]
        final_preds = []

        i = 0
        for j in range(len(flatten_mask)):
            if flatten_mask[j]:
                final_preds.append(flatten_preds[i])
                i += 1
            else:
                final_preds.append(-100)

        num_examples = len(mask)
        seq_size = len(mask[0])
        final_preds = np.array(final_preds).reshape((num_examples, seq_size))
        return final_preds[mask == 1]

    def score(self, preds, true_labels):
        pass
        return 1


# --------------------------------------------------------------------------------------------------------
class RelationClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels=2):
        super(RelationClassificationHead, self).__init__()
        self.hidden_size = hidden_size  # last hidden state of BERT model
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(2 * 768, num_labels)
        self.activation = nn.Sigmoid()

    def forward(self, inputs, batch=None):
        """
        input_ids: shape - (batch_size, seq_len)
        e1_start: the position of the token [E1_start] in the tokinized sentence ([E1_start] is coming right before the first entity)
        e2_start: the position of the token [E2_start] in the tokinized sentence ([E2_start] is coming right before the second entity)
        """
        e1_start = batch['e1_start_pos']
        e2_start = batch['e2_start_pos']
        last_hidden_state_vectors = inputs["last_hidden_state"]
        batch_size = last_hidden_state_vectors.size(0)

        e1_start_embedding = last_hidden_state_vectors[torch.arange(batch_size), e1_start,
                             :]  # shape: [batch_size, 768]
        e2_start_embedding = last_hidden_state_vectors[torch.arange(batch_size), e2_start,
                             :]  # shape: [batch_size, 768]
        joint_embedding = torch.cat((e1_start_embedding, e2_start_embedding), 1)  # shape: [batch_size, 2*768]
        joint_embedding = self.dropout(joint_embedding)
        logits = self.classifier(joint_embedding)  # shape: [batch_size, num_labels=2]
        # scores = self.activation(logits)
        return logits

    def predict(self, input_ids, attention_mask, e1_start, e2_start):
        """
        Returns prediction.
        tensor outut shape: [batch_size] (a 1D tensor of length batch_size)
        """
        with torch.no_grad():
            scores = self.forward(input_ids, attention_mask, e1_start, e2_start)
        preds = torch.argmax(scores, axis=-1)
        return preds
