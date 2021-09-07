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

    # The labels are read from the txt file listing the labels
    data = pd.read_csv('labels.txt', delimiter=',')
    labels = data.columns.to_list()

    # Defining the model
    bert_model = torch.load(fine_tuned_model)

    # Defining the tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model,
                                                    model_max_length=512,
                                                    max_len=512,
                                                    truncation=True,
                                                    padding='Longest'
                                                    )

    # Initializing the trainer
    trainer = Trainer(model=bert_model, tokenizer=bert_tokenizer)

    # Saving the terminal input as input_text
    input_text = input('Enter the discharge summary: ')

    # Predicting the ICD code(s) of the input text. Outputs a list where the first item of the list is a list of the length of number of ICD codes where each element is 0/1 representing if the label is present (1) or not (0)
    predictions = trainer.evaluate(X=[input_text], batch_size=batch_size, thres=threshold, num_labels=len(labels))

    # Creating an empty list to put the ICD code(s) associated with the discharge summary
    prediction_list = []

    # For each prediction and 
    for prediction, label in zip(predictions[0], labels):
        if prediction == 1:
            prediction_list.append(label)

    if all(prediction == 0 for prediction in predictions[0]):
        print('Sorry, no groups were able to be identified')

    else:
        print('The discharge summary belongs to the group(s)', ', '.join(str(group) for group in prediction_list))

