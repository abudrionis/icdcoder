'''
Script to build model

Author: Anastasios Lamproudis

'''

import torch.nn
from torch import nn
from transformers import AutoModel, Trainer


class Model(nn.Module):
    def __init__(self, num_labels:int, path:str):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.linear_layer = nn.Linear(in_features = 768, out_features = num_labels)
        self.ReLU = nn.ReLU()
        self.Dropout_layer = nn.Dropout(p = .1)
        self.transformer = AutoModel.from_pretrained(path)

    def forward(self, X, attention_mask):
        X = self.transformer(X, attention_mask)
        X = X[0][:, 0, :]
        X = self.ReLU(X)
        X = self.Dropout_layer(X)
        X = self.linear_layer(X)
        return X


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(logits.view(-1, self.model.config.num_labels),
                             labels.float().view(-1, self.model.config.num_labels)
                             )
        return (loss, outputs) if return_outputs else loss
