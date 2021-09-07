'''
Script containing fine tuning and evaluation function activated when main argument -train_and_test is used

Author: Sonja Remmer

'''

from scripts.preprocess import read_csv_onehot
from transformers import AutoTokenizer
from Utilities import Trainer, KFoldCrossVal
from Model import Model
from sklearn.model_selection import train_test_split


def finetune_evaluate(train_and_test_data, 
            pre_trained_model,
            new_fine_tuned_model,
            test_size, 
            n_kfold,
            random_state,
            n_epochs,
            batch_size,
            gradient_accumulation,
            learning_rate,
            threshold,
            warm_up):
    

    print(pre_trained_model)
    
    # Reading data
    X, Y = read_csv_onehot(filepath = train_and_test_data)

    # Splitting the data into training data and held out-test set which is not used during the training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                      test_size = test_size,
                                                      shuffle = True)

    # Defining the model
    bert_model = Model(path = pre_trained_model, num_labels = len(Y[0]))
    # Defining the tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model, model_max_length = 512,
        max_len = 512,
        truncation = True, padding='Longest')
    # Initializing the trainer
    trainer = Trainer(model = bert_model, tokenizer = bert_tokenizer)

    # If n_kfold is 1, the training data is not divided into folds am
    if n_kfold == 1:

        trainer.train(
            X = X_train,
            Y = Y_train,
            epochs = n_epochs,
            batch_size = batch_size,
            gradient_accumulation = gradient_accumulation,
            learning_rate = learning_rate,
            thres = threshold,
            warm_up = warm_up,
            save_path = new_fine_tuned_model)
    else: 

        kfold = KFoldCrossVal(nfolds = n_kfold, trainer = trainer, RANDOM_STATE = random_state)
        kfold.train(
            X = X_train,
            Y = Y_train,
            epochs = n_epochs,
            batch_size = batch_size,
            gradient_accumulation = gradient_accumulation,
            learning_rate = learning_rate,
            thres = threshold,
            warm_up = warm_up,
            save_path = new_fine_tuned_model)


