'''
Script containing evaluation function activated when main argument -test is used

Author: Sonja Remmer

'''

import pandas as pd

def read_csv_onehot(filepath):
    '''
    Reads data from csv file and returns list of texts and list of tags
    '''

    # Reading data
    print('\nReading data')
    data = pd.read_csv(filepath, delimiter=',')
    print('\nOriginal dataset\n', data)

    # The discharge summaries should be placed in the second column
    X = data[data.columns[1]].values.tolist()
    # The labels should be one-hot encoded and placed in the third to the last column
    Y = data[data.columns[2:]].values.tolist()

    return X,Y