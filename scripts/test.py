'''
Script containing test function activated when main argument -test is used

Author: Sonja Remmer

'''

from icdcoder.scripts.preprocess import preprocess_csv_baseline
from matplotlib.pyplot import contour
from scripts.preprocess import preprocess_csv_bert, preprocess_text_baseline
from transformers import AutoTokenizer
from Utilities import Trainer
import torch
import pandas as pd
import pickle
from sklearn.metrics import classification_report

def present_predictions(predictions):

    # The labels are read from the txt file listing the labels
    data = pd.read_csv('labels.txt', delimiter=',')
    labels = data.columns.to_list()

    # Creating an empty list to put the ICD code(s) associated with the discharge summary
    prediction_list = []

    # For each prediction and each label, if a prediction is 1, the corresponding label is appended to the list of predictions
    for prediction, label in zip(predictions[0], labels):
        if prediction == 1:
            prediction_list.append(label)

    # If all predictions are 0, it is printed that no ICD codes were predicted (meaning all predictions were below the threshold)
    if all(prediction == 0 for prediction in predictions[0]):
        print('Sorry, no groups were able to be identified')

    # Otherwise, the list of predictions is printed
    else:
        print('The discharge summary belongs to the group(s)', ', '.join(str(group) for group in prediction_list))

def initialize_bert_trainer(pre_trained_model, fine_tuned_model):

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

    return trainer


def test_bert_filepath(test_data, pre_trained_model, fine_tuned_model, batch_size_test, threshold):

    trainer = initialize_bert_trainer(pre_trained_model=pre_trained_model, fine_tuned_model=fine_tuned_model)
        
    # Reading data
    X_test, Y_test = preprocess_csv_bert(filepath = test_data)

    print('\nTesting on all of the dataset\n')

    # The held-out test set is used for evaluation
    predictions = trainer.evaluate(X = X_test, batch_size = batch_size_test, thres = threshold, num_labels = len(Y_test[0]))
    print('\n_____________________________________________________________________________\n')
    print('\nResults for testing on specified dataset\n')
    print(classification_report(Y_test, predictions, zero_division=False))


def test_bert_text(test_text, pre_trained_model, fine_tuned_model, batch_size_test, threshold):

    trainer = initialize_bert_trainer(pre_trained_model=pre_trained_model, fine_tuned_model=fine_tuned_model)

    data = pd.read_csv('labels.txt', delimiter=',')
    labels = data.columns.to_list()

    # Predicting the ICD code(s) of the input text. Outputs a list where the first item of the list is a list of the length of number of ICD codes where each element is 0/1 representing if the label is present (1) or not (0)
    predictions = trainer.evaluate(X = [test_text], batch_size = batch_size_test, thres = threshold, num_labels = len(labels))

    present_predictions(predictions)


def load_baseline_model(trained_model, trained_vectorizer):

    loaded_model = pickle.load(open(trained_model, 'rb'))
    loaded_vectorizer = pickle.load(open(trained_vectorizer, 'rb'))

    return loaded_model, loaded_vectorizer


def test_baseline_filepath(test_data, trained_model, vectorizer, stopwords):

    loaded_model, loaded_vectorizer = load_baseline_model(trained_model = trained_model, trained_vectorizer = vectorizer)

    # Reading data
    X_test, Y_test = preprocess_csv_baseline(filepath = test_data, filepath_stopwords = stopwords)

    print('\nTesting on all of the dataset\n')

    X_test = loaded_vectorizer.transform(X_test)

    # The held-out test set is used for evaluation
    predictions = loaded_model.predict(X_test)
    
    print('\n_____________________________________________________________________________\n')
    print('\nResults for testing on specified dataset\n')
    print(classification_report(Y_test, predictions, zero_division=False))


def test_baseline_text(test_text, trained_model, vectorizer, stopwords):

    loaded_model, loaded_vectorizer = load_baseline_model(trained_model = trained_model, trained_vectorizer = vectorizer)

    text = preprocess_text_baseline(text=test_text, filepath_stopwords=stopwords)

    X_test = [text]
    X_test = loaded_vectorizer.transform(X_test)

    predictions = loaded_model.predict(X_test)

    # Dealing with the fact that MLKNN() outputs a different format
    try:
        if predictions.getformat() == 'lil':
            predictions = loaded_model.predict(X_test).toarray()

    finally: 
        present_predictions(predictions)
