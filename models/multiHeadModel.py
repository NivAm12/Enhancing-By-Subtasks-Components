import torch.nn as nn
import torch


class MultiHeadModel(nn.Module):
    """
    Multi-Head Model for performing multiple tasks using a shared base model.

    Args:
        base_model (nn.Module): The pre-trained base model used for feature extraction.
        classifier_heads (nn.ModuleDict): A dictionary containing classifier heads for different tasks.
                                 Keys are head names (strings) and values are classifier modules (nn.Module).

    Example:
        base_model = BertModel.from_pretrained('bert-base-uncased')
        classifier_heads = {'sentiment': SentimentClassifier(), 'ner': NERClassifier()}
        multi_head_model = MultiHeadModel(base_model, classifier_heads)

        # For inference/testing
        output = multi_head_model(input_tokens, head_to_use='sentiment')
    """
    def __init__(self, base_model: nn.Module, classifier_heads: nn.ModuleDict):
        super(MultiHeadModel, self).__init__()
        self.base_model = base_model
        self.heads = classifier_heads

    def forward(self, tokens, head_to_use: str):
        outputs = self.base_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
                                  token_type_ids=tokens['token_type_ids'])
        outputs = self.heads[head_to_use](outputs)

        return outputs