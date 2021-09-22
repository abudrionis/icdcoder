'''
Script containing train and test function activated when main argument -train_and_test is used

Author: Sonja Remmer

'''

from scripts.preprocess import preprocess_csv_bert, preprocess_csv_baseline
from transformers import AutoTokenizer
from Utilities import Trainer, KFoldCrossVal
from Model import Model
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, f1_score, make_scorer
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pickle
import time


def train_test_bert(train_and_test_data, 
            pre_trained_model,
            new_fine_tuned_model,
            test_size, 
            n_kfold,
            random_state,
            n_epochs,
            batch_size_train,
            batch_size_test,
            gradient_accumulation,
            learning_rate,
            threshold,
            warm_up):

    # Setting the random state
    np.random.seed(random_state)
    
    # Reading data
    X, Y = preprocess_csv_bert(filepath = train_and_test_data)

    # Splitting the data into training data and held out-test set which is not used during the training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                      test_size = test_size,
                                                      shuffle = True,
                                                      random_state=random_state)

    # Defining the model
    bert_model = Model(path = pre_trained_model, num_labels = len(Y[0]))
    # Defining the tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model, model_max_length = 512,
        max_len = 512,
        truncation = True, padding='Longest')
    # Initializing the trainer
    trainer = Trainer(model = bert_model, tokenizer = bert_tokenizer)

    # If n_kfold is 1 or less, the training data is not divided into folds and all training data is used for training
    if n_kfold <= 1:

        # Splitting the training data into a new training set and a validation set
        X_train_new, X_val, Y_train_new, Y_val = train_test_split(X_train, Y_train,
                                                      test_size = 0.1,
                                                      random_state=random_state,
                                                      shuffle = True)

        # Training and saving the model
        print('\nTraining on all of the training data\n')
        start_time = time.time()
        trainer.train(
            X = X_train_new,
            Y = Y_train_new,
            X_val = X_val,
            Y_val = Y_val,
            epochs = n_epochs,
            batch_size = batch_size_train,
            gradient_accumulation = gradient_accumulation,
            learning_rate = learning_rate,
            thres = threshold,
            warm_up = warm_up,
            save_path = new_fine_tuned_model)
    
        # The held-out test set is used for evaluation
        print('\nEvaluating the fine-tuned model on the held-out test set of size', test_size, '\n')
        predictions = trainer.evaluate(X=X_test, batch_size=batch_size_test, thres=threshold, num_labels = len(Y[0]))
        print('\n_____________________________________________________________________________\n')
        print('\nResults for BERT classifier trained on all training data and tested on held-out test set\n')
        print(classification_report(Y_test, predictions, zero_division=False))
        print('\n--- Training on all the training data and testing on the held-out test set using the BERT model took %s seconds ---' % (time.time() - start_time), '\n')

    # If n_kfold is more than 1, the training data is divided into n_kfold number of parts which are subject for k-fold cross-validation. The held-out set is left untouched.
    
    else:
        print('\nDoing k-fold cross-validation with', n_kfold, 'number of folds. The held-out test set of size', test_size, 'is left untouched\n')
        start_time = time.time()
        kfold = KFoldCrossVal(nfolds = n_kfold, trainer = trainer, RANDOM_STATE = random_state)
        kfold.train(
            X = X_train,
            Y = Y_train,
            epochs = n_epochs,
            batch_size = batch_size_train,
            gradient_accumulation = gradient_accumulation,
            learning_rate = learning_rate,
            thres = threshold,
            warm_up = warm_up,
            save_path = new_fine_tuned_model)
        print('\n--- Cross-validation of the training set using the BERT model took %s seconds ---' % (time.time() - start_time), '\n')


def train_test_baseline(train_and_test_data, 
                stopwords,
                new_vectorizer,
                new_trained_model, 
                test_size, 
                random_state, 
                classifier,
                n_kfold):

    # Setting the random state
    np.random.seed(random_state)

    # Reading preprocessed data
    X, Y = preprocess_csv_baseline(filepath = train_and_test_data, filepath_stopwords = stopwords)

    # Splitting data into train and held-out test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state, shuffle = True)

    # Creating an instantiation of the tf-idf vectorizer, fitting it to the discharge summaries
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(X_train)

    # Saving the vectorizer
    pickle.dump(vectorizer, open(new_vectorizer, 'wb'))

    # Creating the tf-idf representation of the train and test sets
    X_train_vectors = vectorizer.transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Instantiating the chosen classifier. Using default hyper-parameters
    if classifier == 'KNN':
        chosen_classifier = MLkNN()
    elif classifier == 'DT':
        chosen_classifier = DecisionTreeClassifier(random_state = random_state)
    elif classifier == 'SVM':
        chosen_classifier = OneVsRestClassifier(SVC(random_state = random_state))

    # If n_kfold is 1 or less, the training data is not divided into folds and all training data is used for training
    if n_kfold <= 1:

        start_time = time.time()
        # Fitting the chosen classifier to the whole training set
        chosen_classifier.fit(X_train_vectors, Y_train)

        # Saving the trained model
        pickle.dump(chosen_classifier, open(new_trained_model, 'wb'))
        
        print('\n_____________________________________________________________________________\n')
        print('\nResults for classifier', chosen_classifier, 'trained on all training data and tested on held-out test set\n')
        # Using the classifier to predict the held-out test set
        predictions = chosen_classifier.predict(X_test_vectors)
        print(classification_report(Y_test, predictions, zero_division = False))
        print('\n--- Training on all the training data and testing on the held-out test set using', chosen_classifier, 'took %s seconds ---' % (time.time() - start_time), '\n')

    # If n_kfold is more than 1, the training data is divided into n_kfold number of parts which are trained seperatley 
    else:    

        # Defining an evaluation function to use as scoring argument in cross_val_score function. This is done to be able to evaluate each fold seperatly and combined
        def evaluation_per_fold(y_true, y_pred):
            # Putting the true classes of each fold in the trueclass list
            trueclass.extend(y_true)
            # If the classifier is MLKNN, the predicted classes of each fold are transformed to an array before put in the predictedclass list (only needed for MLKNN)
            if classifier == 'KNN':
                predictedclass.extend(y_pred.toarray())
            # For all the other classififers, the predicted classes of each fold are put directly in the predictedclass list
            else:
                predictedclass.extend(y_pred)
            # Printing classification reports per fold
            print(classification_report(y_true, y_pred, zero_division = False))
            # The F1-micro of each fold is calculated and appended to the f1 list
            f1score = f1_score(y_true, y_pred, average = 'micro')
            f1.append(f1score)
            return f1score        

        # Creating a k-fold instantiation (data is shuffled)
        k_fold = KFold(n_splits = n_kfold, random_state = random_state, shuffle = True)

        print('\n_____________________________________________________________________________\n')
        print('\nRESULTS FOR', chosen_classifier,'\n')
        print('\n__________________________\n')
        print('\nResults per fold for', chosen_classifier, '\n')
        # Creating empty lists to put the true classes, the predicted classes, and the F1-micro scores for all the folds in
        trueclass = []
        predictedclass = []
        f1 = []
        # Saving the starting time of the cross-validation
        start_time = time.time()
        # Calculating evaluation metrics for each fold using the training data, the k-fold instantiation and the custom evaluation per fold function
        f1score = cross_val_score(chosen_classifier, X_train_vectors, Y_train, cv = k_fold, scoring = make_scorer(evaluation_per_fold))
        # Calculating the time it took to perform cross-validation
        print('\n--- Cross-validation of the training set using', chosen_classifier, 'took %s seconds ---' % (time.time() - start_time), '\n')
        print('\n__________________________\n')
        print('\nCombined results for all folds for', chosen_classifier, '\n')
        # Printing a classification report for all folds combined (full training set)
        print(classification_report(trueclass, predictedclass, zero_division = False))
        print('\nF1-micro for each fold of the data for', chosen_classifier, f1)
        # Using the lists that have collected all the folds' true and predicted classes to calculate the F1-micro for all the folds combined
        f1_combined = f1_score(trueclass, predictedclass, average = 'micro')

