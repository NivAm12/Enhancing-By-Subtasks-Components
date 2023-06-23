import torch.nn as nn


class MultiHeadModel(nn.Module):
    def __init__(self, base_model: nn.Module, classifier_heads: list):
        super(MultiHeadModel, self).__init__()
        self.base_model = base_model
        self.heads = classifier_heads

    def forward(self, tokens, head_to_use):
        outputs = self.base_model(**tokens)
        outputs = self.heads[head_to_use](outputs)

        return outputs