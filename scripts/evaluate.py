'''
Script containing evaluation function activated when main argument -test is used

Author: Sonja Remmer
'''

from transformers import AutoTokenizer
from Utilities import Trainer
import torch
import pandas as pd
import numpy as np


def evaluate(pre_trained_model, fine_tuned_model, batch_size, threshold):


    data = pd.read_csv('labels.txt', delimiter=',')
    labels = data.columns.to_list()

    bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model,
                                                    model_max_length=512,
                                                    max_len=512,
                                                    truncation=True,
                                                    padding='Longest'
                                                    )

    bert_model = torch.load(fine_tuned_model)

    trainer = Trainer(model=bert_model, tokenizer=bert_tokenizer)

    input_text = input('Enter the discharge summary: ')

    predictions = trainer.evaluate(X=[input_text], batch_size=batch_size, thres=threshold, num_labels=len(labels))

    prediction_list = []

    for prediction, label in zip(predictions[0], labels):
        if prediction == 1:
            prediction_list.append(label)

    if all(prediction == 0 for prediction in predictions[0]):
        print('Sorry, no groups were able to be identified')

    else:
        print('The discharge summary belongs to the group(s)', ', '.join(str(group) for group in prediction_list))

