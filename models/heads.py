import torch.nn as nn
import torch


class ClassificationHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.mean = torch.mean
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x[0])
        x = self.mean(x, dim=1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x