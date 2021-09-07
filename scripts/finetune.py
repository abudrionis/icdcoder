'''
Script containing finetuning function activated when main argument -train is used

Author: Sonja Remmer

'''

from scripts.preprocess import read_csv_onehot
from transformers import AutoTokenizer
from Utilities import Trainer
from Model import Model
from sklearn.model_selection import train_test_split


def finetune(train_data, 
            pre_trained_model,
            new_fine_tuned_model,
            n_epochs,
            batch_size,
            gradient_accumulation,
            learning_rate,
            threshold,
            warm_up):

    # Reading data
    X, Y = read_csv_onehot(filepath=train_data)

    # Splitting data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      test_size = 0.1,
                                                      shuffle = True)

    # Defining the model
    bert_model = Model(path = pre_trained_model, num_labels = len(Y[0]))
    # Defining the tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model, model_max_length = 512,
        max_len = 512,
        truncation = True, padding = 'Longest')
    # Initializing the trainer
    trainer = Trainer(model = bert_model, tokenizer = bert_tokenizer)

    # Starting the training
    trainer.train(
        X = X_train,
        Y = Y_train,
        X_val = X_val,
        Y_val = Y_val,
        epochs = n_epochs,
        batch_size = batch_size,
        gradient_accumulation = gradient_accumulation,
        learning_rate = learning_rate,
        thres = threshold,
        warm_up = warm_up,
        save_path = new_fine_tuned_model)





