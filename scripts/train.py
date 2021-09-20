'''
Script containing train function activated when main argument -train is used

Author: Sonja Remmer

'''

from scripts.preprocess import preprocess_csv_bert, preprocess_csv_baseline
from transformers import AutoTokenizer
from Utilities import Trainer
from Model import Model
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pickle
np.set_printoptions(threshold=np.inf)



def train_bert(train_data, 
            pre_trained_model,
            new_fine_tuned_model,
            n_epochs,
            batch_size_train,
            random_state,
            gradient_accumulation,
            learning_rate,
            threshold,
            warm_up):
            

    # Reading data
    X, Y = preprocess_csv_bert(filepath=train_data)

    # Splitting data into training and validation sets. Note that this validation set is used during training and is not the held-out test set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      test_size = 0.1,
                                                      random_state=random_state,
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

    # Training and saving the model
    trainer.train(
        X = X_train,
        Y = Y_train,
        X_val = X_val,
        Y_val = Y_val,
        epochs = n_epochs,
        batch_size = batch_size_train,
        gradient_accumulation = gradient_accumulation,
        learning_rate = learning_rate,
        thres = threshold,
        warm_up = warm_up,
        save_path = new_fine_tuned_model)


def train_baseline(train_data, 
            stopwords, 
            new_vectorizer,
            new_trained_model, 
            random_state, 
            classifier):

    # Setting the random state
    np.random.seed(random_state)

    # Reading and preprocessing the data. All the data is used for training 
    X_train, Y_train = preprocess_csv_baseline(filepath=train_data, filepath_stopwords=stopwords)

    # Creating an instantiation of the tf-idf vectorizer, fitting it to the discharge summaries
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(X_train)

    # Saving the vectorizer
    pickle.dump(vectorizer, open(new_vectorizer, 'wb'))

    # Vectorizing the training data
    X_train_vectors = vectorizer.transform(X_train)

    # Instantiating the chosen classifier. Default hyper-parameters are used
    if classifier == 'KNN':
        chosen_classifier = MLkNN()
    elif classifier == 'DT':
        chosen_classifier = DecisionTreeClassifier(random_state=random_state)
    elif classifier == 'SVM':
        chosen_classifier = OneVsRestClassifier(SVC(random_state=random_state))

    # Fitting the chosen classifier to the training data
    chosen_classifier = chosen_classifier.fit(X_train_vectors, Y_train)

    # Saving the trained model
    pickle.dump(chosen_classifier, open(new_trained_model, 'wb'))






