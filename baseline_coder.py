'''
Main script for using the baseline models

Author: Sonja Remmer

'''

from scripts.train import train_baseline
from scripts.train_test import train_test_baseline
from scripts.test import test_baseline

import argparse
import os


def main(train_and_test_data,
    train_data, 
    test, 
    trained_model,
    vectorizer,
    new_vectorizer,
    new_trained_model,
    test_size,
    n_kfold, 
    random_state,
    classifier,
    stopwords):
    
    # If argument train_and_test_data is given, the finetune_evaluate function is run
    if train_and_test_data:
        train_test_baseline(train_and_test_data, 
        stopwords, 
        test_size, 
        random_state, 
        classifier,
        n_kfold)

    # If argument train_data is given, the finetune function is run
    elif train_data:
        train_baseline(train_data, 
            stopwords, 
            new_vectorizer,
            new_trained_model, 
            random_state, 
            classifier)

    # If argument test is given, the evaluate function is run
    elif test:
        test_baseline(trained_model, 
        vectorizer, 
        stopwords)


if __name__ == '__main__':

    # Defining default values
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_stopwords = base_dir+'/stopwords.txt'
    default_trained_model = base_dir+'/models/trained_baseline_model/ICD_model.sav'
    default_vectorizer = base_dir+'/models/trained_baseline_model/vectorizer.sav'
    default_new_trained_vectorizer = base_dir+'/models/new_baseline_model/vectorizer.sav'
    default_new_trained_model = base_dir+'/models/new_baseline_model/ICD_model.sav'
    default_test_size = 0.1
    default_random_state = None
    default_classifier = 'SVM'
    default_kfold = 5
    
    
    parser = argparse.ArgumentParser()
    mutually_exclusive = parser.add_mutually_exclusive_group(required=True)

    # Arguments -train, -train_and_test and -test are mutually exclusive and at least one is required
    mutually_exclusive.add_argument('-train', dest='only_train_data', type=str, default=None,
                        help='Filepath to csv file used for training. The file needs to follow the structure specified in the README section How to prepare dataset used for fine-tuning.')
    mutually_exclusive.add_argument('-train_and_test', dest='train_and_test_data', type=str, default=None,
                        help='Filepath to csv file used for training and testing. The file needs to follow the structure specified in the README section How to prepare dataset used for fine-tuning.')
    mutually_exclusive.add_argument('-test', dest='only_test_data', action='store_true', default=None,
                        help='Use argument if you want to predict the ICD codes of an unseen discharge summary')
    
    # The rest of the arguments are optional and while not all functions are compatible with all main arguments, passing a non-compatible argument will just be ignored 
    parser.add_argument('-stopwords', dest='stopwords', type=str, default=default_stopwords,
                        help='Filepath to stopwords. Default is subfolder ./stopwords.txt', required=False)
    parser.add_argument('-trained_model', dest='trained_model', type=str, default=default_trained_model,
                        help='Filepath to traind model. Note that if trained model is specified, the vectorizer needs to be specified as well. Default is ./models/trained_baseline/ICD_model', required=False)
    parser.add_argument('-vectorizer', dest='vectorizer', type=str, default=default_vectorizer,
                        help='Filepath to traind vectorizer. Note that if vectorizer is specified, the trained model needs to be specified as well.  Default is ./models/trained_baseline/vectorizer.sav', required=False)
    parser.add_argument('-new_trained_model', dest='new_trained_model', type=str, default=default_new_trained_model,
                        help='Filepath to new traind model. Default is ./models/trained_baseline/ICD_model', required=False)
    parser.add_argument('-new_trained_vectorizer', dest='new_vectorizer', type=str, default=default_new_trained_vectorizer,
                        help='Filepath to new trained vectorizer. Default is ./models/trained_baseline/vectorizer.sav', required=False)
    parser.add_argument('-test_size', dest='test_size', type=str, default=default_test_size,
                        help='Fraction of data to use for testing. Must be between 0 and 1. Default is 0.1.', required=False)
    parser.add_argument('-random_state', dest='random_state', type=int, default=default_random_state,
                        help='A seed (integer) to use as the random state in the k-fold cross-validation. Default is None.', required=False)
    parser.add_argument('-kfold', dest='n_kfold', type=int, required=False, default=default_kfold,
                        help='The number of folds (k) to use in k-fold cross-validation, must be > 1 for kfold to be used and default is 5.')
    parser.add_argument('-classifier', dest='classifier', type=str, required=False, default=default_classifier,
                        help='The desired classifier. Enter SVM, KNN or DT which represent Support Vector Machines, K-Nearest Neigbors, and Decision Trees. Default is SVM.')
    
    args = parser.parse_args()

    main(train_and_test_data=args.train_and_test_data,
        train_data=args.only_train_data,
        test=args.only_test_data,
        stopwords=args.stopwords,
        trained_model=args.trained_model,
        vectorizer=args.vectorizer,
        new_vectorizer=args.new_vectorizer,
        new_trained_model=args.new_trained_model,
        test_size=args.test_size,
        random_state=args.random_state,
        n_kfold=args.n_kfold,
        classifier=args.classifier)
