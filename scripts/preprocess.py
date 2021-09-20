'''
Script containing evaluation function activated when main argument -test is used

Author: Sonja Remmer

'''

import pandas as pd

def preprocess_csv_bert(filepath):
    '''
    Reads data from csv file and returns list of texts and list of tags
    '''

    # Reading data
    print('\nReading data')
    data = pd.read_csv(filepath, delimiter=',')
    print('\nFirst rows of dataset\n', data.head(10))

    # The discharge summaries should be placed in the second column
    X = data[data.columns[1]].values.tolist()
    # The labels should be one-hot encoded and placed in the third to the last column
    Y = data[data.columns[2:]].values.tolist()

    return X,Y


def preprocess_csv_baseline(filepath, filepath_stopwords):
    '''
    Reads data from csv file, does basic preprocessing and returns list of texts and list of tags
    '''

    # Reading data
    print('\nReading data')
    data = pd.read_csv(filepath, delimiter=',')
    print('\nFirst rows of original dataset\n\n', data.head(10))

    # Decapitalizing the discharge summaries and stripping thme from punctuation and stopwords
    data[data.columns[1]] = data[data.columns[1]].str.lower()
    data[data.columns[1]] = data[data.columns[1]].str.replace(r'[^\w\s]', '', regex=True)
    with open(filepath_stopwords, 'r') as f:
        stop_words = f.readlines()
    stop_words = stop_words[0]
    data[data.columns[1]] = data[data.columns[1]].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))
    print('\n__________________________\n')
    print('\First rows of dataset without upper case letters, punctuation, and stopwords\n\n', data.head(10))

    # The discharge summaries should be placed in the second column
    X = data[data.columns[1]].to_numpy()
    # The labels should be one-hot encoded and placed in the third to the last column
    Y = data[data.columns[2:]].to_numpy()

    return X,Y

def preprocess_text_baseline(text, filepath_stopwords):

    text = text.lower()
    text = text.replace(r'[^\w\s]', '')
    textwords = text.split()

    with open(filepath_stopwords, 'r') as f:
        stop_words = f.readlines()
    stop_words = stop_words[0]
    textwords_clean  = [word for word in textwords if word.lower() not in stop_words]
    text = ' '.join(textwords_clean)

    return text