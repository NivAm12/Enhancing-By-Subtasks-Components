import torch.nn as nn
import torch
from TorchCRF import CRF


class ClassificationHead(nn.Module):
    """
    A classification head for a language model.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    """
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.2, inplace=False)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = None if out_features == 1 else nn.Softmax()

    def forward(self, inputs):
        # outputs = self.dropout(inputs.pooler_output)
        outputs = self.linear(inputs.pooler_output) # pooler_output is the index of cls token
        outputs = outputs if self.activation is None else self.activation(outputs)

        return outputs


class BERT_CRF(nn.Module):

    def __init__(self, bert_model, num_labels):
        super(BERT_CRF, self).__init__()
        self.hidden_size = 768 # last hidden state of BERT model
        self.bert = bert_model
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(768, num_labels)
        self.activation = nn.ReLU()
        self.crf = CRF(num_labels) #, batch_first = True)

    def forward(self, input_ids, attention_mask,  labels=None, token_type_ids=None):
        """
        input_ids: shape - (batch_size, seq_len)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # shape: (batch_size, seq_len, embedding_dim)
        embeddings = outputs["last_hidden_state"]
        embeddings = self.dropout(embeddings)
        emission_scores = self.classifier(embeddings) # shape: (batch_size, seq_len, num_labels)
        emission_scores = self.activation(emission_scores)
        #if labels:
        raw_prediction = self.crf.forward(emission_scores, labels, attention_mask)
        #else:
        #   raw_prediction = self.crf.viterbi_decode(emission_scores, mask=attention_mask.type(torch.uint8))
        return raw_prediction

        # emission = self.classifier(sequence_output) # [32,256,17]
        # labels=labels.reshape(attention_mask.size()[0],attention_mask.size()[1])

        # if labels is not None:
        #     loss = -self.crf(log_soft(emission, 2), labels, mask=attention_mask.type(torch.uint8), reduction='mean')
        #     prediction = self.crf.decode(emission, mask=attention_mask.type(torch.uint8))
        #     return [loss, prediction]

        # else:
        #     prediction = self.crf.decode(emission, mask=attention_mask.type(torch.uint8))
        #     return prediction
