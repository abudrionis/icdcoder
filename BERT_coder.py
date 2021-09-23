'''
Main script for using BERT

Author: Sonja Remmer

'''

import argparse
import os

from scripts.train import train_bert
from scripts.test import test_bert_filepath, test_bert_text
from scripts.train_test import train_test_bert


def main(train_and_test_data,
        train_data, 
        test_data,
        test_text, 
        pre_trained_model,
        fine_tuned_model,
        new_fine_tuned_model,
        test_size,
        n_kfold, 
        random_state,
        n_epochs,
        batch_size_train,
        batch_size_test,
        gradient_accumulation,
        learning_rate,
        warm_up,
        threshold):
    
    # If argument train_and_test_data is given, the finetune_evaluate function is run
    if train_and_test_data:
        train_test_bert(train_and_test_data=train_and_test_data, 
            pre_trained_model=pre_trained_model,
            new_fine_tuned_model=new_fine_tuned_model,
            test_size=test_size, 
            n_kfold=n_kfold,
            random_state=random_state,
            n_epochs=n_epochs,
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            gradient_accumulation=gradient_accumulation,
            learning_rate=learning_rate,
            warm_up=warm_up,
            threshold=threshold)

    # If argument train_data is given, the finetune function is run
    elif train_data:
        train_bert(train_data=train_data, 
            pre_trained_model=pre_trained_model,
            new_fine_tuned_model=new_fine_tuned_model,
            n_epochs=n_epochs,
            batch_size_train=batch_size_train,
            random_state=random_state,
            gradient_accumulation=gradient_accumulation,
            learning_rate=learning_rate,
            warm_up=warm_up,
            threshold=threshold)
    
    # If argument test_data is given, the evaluate function is run with the test filepath as input
    elif test_data:
        test_bert_filepath(test_data=test_data,
            pre_trained_model=pre_trained_model, 
            fine_tuned_model=fine_tuned_model,
            batch_size_test=batch_size_test,
            threshold=threshold)
    
    elif test_text:
        test_bert_text(test_data=test_text,
            pre_trained_model=pre_trained_model, 
            fine_tuned_model=fine_tuned_model,
            batch_size_test=batch_size_test,
            threshold=threshold)


if __name__ == '__main__':

    # Defining default values
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_pre_trained_model = base_dir+'/models/pre_trained_model'
    default_fine_tuned_model = base_dir+'/models/fine_tuned_model/pytorch_model.bin'
    default_new_fine_tuned_model = base_dir+'/models/new_fine_tuned_model'
    default_test_size = 0.1
    default_kfold = 10  
    default_random_state = None
    default_epochs = 10
    default_batch_size_train = 4
    default_batch_size_test = 2
    default_gradient_accumulation = 8
    default_learning_rate = 2e-5
    default_warm_up = 155
    default_threshold = 0.5
    
    

    parser = argparse.ArgumentParser()
    mutually_exclusive = parser.add_mutually_exclusive_group(required=True)

    # Arguments -train, -train_and_test and -test are mutually exclusive and at least one is required
    mutually_exclusive.add_argument('-train', dest='only_train_data', type=str, default=None,
                        help='Filepath to csv file used for training. The file needs to follow the structure specified in the README section How to get hold of/prepare datasets.')
    mutually_exclusive.add_argument('-train_and_test', dest='train_and_test_data', type=str, default=None,
                        help='Filepath to csv file used for training and testing. The file needs to follow the structure specified in the README section How to get hold of/prepare datasets.')
    mutually_exclusive.add_argument('-test_data', dest='test_data', type=str, default=None,
                        help='Filepath to csv file used for testing. The file needs to follow the structure specified in the README section How to get hold of/prepare datasets.')
    mutually_exclusive.add_argument('-test_text', dest='test_text', type=str, default=None,
                        help='Discharge summary to predict ICD codes for.')
    
    # The rest of the arguments are optional and while not all functions are compatible with all main arguments, passing a non-compatible argument will just be ignored 
    parser.add_argument('-pre_trained', dest='pre_trained_model', type=str, default=default_pre_trained_model,
                        help='Filepath to folder with pre-trained model. Default is subfolder ./models/pre_trained_model', required=False)
    parser.add_argument('-fine_tuned', dest='fine_tuned_model', type=str, default=default_fine_tuned_model,
                        help='Filepath to fine-tuned (traind) model. Default is ./models/fine_tuned_model/pytorch_model.bin', required=False)
    parser.add_argument('-new_fine_tuned', dest='new_fine_tuned_model', type=str, default=default_new_fine_tuned_model,
                        help='Filepath to folder to save new fine-tuned model in', required=False)


    parser.add_argument('-test_size', dest='test_size', type=float, required=False, default=default_test_size,
                        help='Fraction of data to use for testing. Must be between 0 and 1. Default is 0.1.')
    parser.add_argument('-kfold', dest='n_kfold', type=int, required=False, default=default_kfold,
                        help='The number of folds (k) to use in k-fold cross-validation, must be > 1 for kfold to be used and default is 10. If k-fold is used, the held-out test set is not used. If k-fold is not used, testing is done on the held-out test set.')
    parser.add_argument('-random_state', dest='random_state', type=int, default=default_random_state,
                        help='A seed (integer) to use as the random state when splitting the data. Default is None', required=False)


    parser.add_argument('-epochs', dest='n_epochs', type=int, default=default_epochs,
                        help='Number of epochs to train for. Default is 10.', required=False)
    parser.add_argument('-batch_size_train', dest='batch_size_train', type=int, default=default_batch_size_train,
                        help='The batch size for training. Default is 4.', required=False)
    parser.add_argument('-batch_size_test', dest='batch_size_test', type=int, default=default_batch_size_test,
                        help='The batch size for testing. Default is 2.', required=False)
    parser.add_argument('-gradient_accumulation', dest='gradient_accumulation', type=int, default=default_gradient_accumulation,
                        help='The gradient accumulation. The batch size multiplied with the gradient accumulation is the actual batch size. Default is 8.', required=False)
    parser.add_argument('-learning_rate', dest='learning_rate', type=float, default=default_learning_rate,
                        help='The learning rate. Default is 2e-5.', required=False)
    parser.add_argument('-warm_up', dest='warm_up', type=int, default=default_warm_up,
                        help='The number of warm-up steps, that is, the number of steps before the learning rate starts to decay. Default is 155.', required=False)
    parser.add_argument('-threshold', dest='threshold', type=float, default=default_threshold,
                        help='The threshold that binarizes the model output (0: label not present, 1: label present). Should be a number between 0 and 1, default is 0.5.', required=False)

    args = parser.parse_args()

    main(train_and_test_data=args.train_and_test_data,
        train_data=args.only_train_data,
        test_data=args.test_data,
        test_text=args.test_text,
        pre_trained_model=args.pre_trained_model,
        fine_tuned_model=args.fine_tuned_model,
        new_fine_tuned_model=args.new_fine_tuned_model,
        test_size=args.test_size,
        n_kfold=args.n_kfold,
        random_state=args.random_state, 
        n_epochs=args.n_epochs,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warm_up=args.warm_up,
        threshold=args.threshold
        )

    
