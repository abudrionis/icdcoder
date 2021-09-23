'''
Script containing test function activated when main argument -test is used

Author: Sonja Remmer

'''

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


def test_bert(pre_trained_model, fine_tuned_model, batch_size_test, threshold):

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

    while True:

        input_choice = input('\nDo you want to:\n(1) predict the ICD codes of a single discharge summary or\n(2) evaluate the model by entering the filepath to a csv file with test data?\n\nEnter 1 or 2: ')

        if input_choice == '1':

            # Saving the terminal input as input_text
            input_text = input('Enter the discharge summary: ')

            data = pd.read_csv('labels.txt', delimiter=',')
            labels = data.columns.to_list()

            # Predicting the ICD code(s) of the input text. Outputs a list where the first item of the list is a list of the length of number of ICD codes where each element is 0/1 representing if the label is present (1) or not (0)
            predictions = trainer.evaluate(X = [input_text], batch_size = batch_size_test, thres = threshold, num_labels = len(labels))

            present_predictions(predictions)
            break
        
        elif input_choice == '2':

            test_data = input('Enter the filepath to the test data: ')

            # Reading data
            X_test, Y_test = preprocess_csv_bert(filepath = test_data)

            print('\nTesting on all of the dataset\n')

            # The held-out test set is used for evaluation
            predictions = trainer.evaluate(X = X_test, batch_size = batch_size_test, thres = threshold, num_labels = len(Y_test[0]))
            print('\n_____________________________________________________________________________\n')
            print('\nResults for testing on specified dataset\n')
            print(classification_report(Y_test, predictions, zero_division=False))
            break
    
        print('\nYou have to choose alternative 1 or 2\n')
        


def test_baseline(trained_model, vectorizer, stopwords):

    loaded_model = pickle.load(open(trained_model, 'rb'))
    vectorizer = pickle.load(open(vectorizer, 'rb'))

    while True:

        input_choice = input('\nDo you want to:\n(1) predict the ICD codes of a single discharge summary or\n(2) evaluate the model by entering the filepath to a csv file with test data?\n\nEnter 1 or 2: ')

        if input_choice == '1':

            # Saving the terminal input as input_text
            input_text = input('Enter the discharge summary: ')

            text = preprocess_text_baseline(input_text, stopwords)

            X_test = [text]
            X_test = vectorizer.transform(X_test)

            predictions = loaded_model.predict(X_test)#.toarray()

            print(type(predictions[0]))

            present_predictions(predictions)
            break
        
        elif input_choice == '2':

            test_data = input('Enter the filepath to the test data: ')

            # Reading data
            X_test, Y_test = preprocess_csv_bert(filepath = test_data)

            print('\nTesting on all of the dataset\n')

            # The held-out test set is used for evaluation
            predictions = loaded_model.predict(X_test)
            print('\n_____________________________________________________________________________\n')
            print('\nResults for testing on specified dataset\n')
            print(classification_report(Y_test, predictions, zero_division=False))
            break
    
        print('\nYou have to choose alternative 1 or 2\n')
    

