import torch.nn as nn


class MultiHeadModel(nn.Module):
    def __init__(self, base_model: nn.Module, classifier_heads: list):
        super(MultiHeadModel, self).__init__()
        self.base_model = base_model
        self.heads = classifier_heads

    def forward(self, tokens, head_to_use):
        outputs = self.base_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
                                  token_type_ids=tokens['token_type_ids'])
        outputs = self.heads[head_to_use](outputs)

        return outputs