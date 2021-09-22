# Multi-label ICD-10 Classification
*Note that this project was done using Python 3.7.3*

*If you are experiencing issues, the Python version may be the problem.*

## What and why

The purpose of this project is to facilitate:


**(1)** training (fine-tuning) pre-trained BERT models and traditional supervised machine learning models using discharge summaries and diagnosis (ICD) codes, and

**(2)** evaluating how these trained models performs in pairing unseen discharge summaries with the correct ICD codes

## How to get hold of/prepare dataset used for training 

To obtain a valid dataset to use for training the models use one of the two options below.

**(1)** Contact Hercules Dalianis at hercules@dsv.su.se to get hold of the Swedish EPR Gastro ICD-10 Pseudo Corpus containing approximately 6000 Swedish discharge summaries from 5000 patients.  

**(2)** If you want to train the models using your own data, the data needs to be in a csv file adhering to the following format:

***Column 1***: *Patient ID (optional)*


**Column 2**: Discharge summary


**Column 3–Column n**: One ICD code per column with values 0/1 representing whether the code is paired (1) or not (0) with the discharge summary

*n represents the number of ICD codes

*As indicated, patient ID is not necessary, but it is necessary that the discharge summaries are placed in the second column and that the ICD codes are placed in the third to the last colums.*

Below, you can see how the data should be structured

| *Patient ID*  | Discharge summary            | K00-K014    | K20-K31     |  ...         | K90-K93     |
| ------------- | -----------------------------| -----------:| -----------:| ------------:| -----------:|
| *41234131*    | Diagnosed liver cancer trea...| 1           | 0           | ...          | 1           |
| *65464366*    | Pat otherwise healthy came...| 0           | 1           | ...          | 0           |
| ...           | ...                          | ...         | ...         | ...          | ...         |
| *78676876*    | Chrons dis patient sent hom...| 1           | 0           | ...          | 0           |

Moreover, if you have your own dataset with other labels than the Swedish EPR Gastro ICD-10 Pseudo Corpus, you need to replace the already existing *labels.txt* with your own version of the file by listing the ICD codes in the order they appear in the dataset. For example, if your ICD codes consist of the ICD blocks in Chapter XI, the txt file should look like this

`K00-K14,K20-K31,K35-K38,K40-K46,K50-K52,K55-K64,K65-K67,K70-K77,K80-K87,K90-K93`

It is important that your file is named *labels.txt* and placed in the same place as the current file. 

## How to get started 


**(1)** Download this repository by opening the terminal/command prompt and changing your directory to where you want to put the project 

`cd [filepath to where I want to put the project]`

For example, creating a folder on the desktop named My-ICD-Project and entering

`cd Desktop/My-ICD-Project`

Then, you clone the project by entering

`git clone https://github.com/sonjaremmer/icdcoder.git`

*Note: when asked to enter your GitHub password, you should instead enter a personal access token which can be created by following these instructions: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token*


**(2)** If you want to use BERT to do the ICD classification, download the BERT model pre-trained on Swedish texts, KB-BERT (bert-base-swedish-cased), by clicking on this link: https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/pytorch_model.bin Put the *pytorch_model.bin* file in the folder *icdcoder/models/pre_trained_model*. It is important that the model file is named *pytorch_model.bin*. 

If you want another pre-trained model, you need to put the associated *config.json* and *vocab.txt* file in the *icdcoder/models/pre_trained_model* folder as well. This is not needed if you are working with *bert-base-swedish-cased* since those config and vocal files are already in the project folder. 


**(3)** If you want to access a KB-BERT model fine-tuned on pseudonymized Swedish discharge summaries (the Stockholm EPR Gastro ICD-10 Pseudo Corpus version 2), contact Hercules Dalianis at hercules@dsv.su.se and put the pytorch_model.bin file in the folder *icdcoder/models/fine_tuned_model*. It is important that the model file is named *pytorch_model.bin*.


**(4)** Change the directory to the icdcoder repository by going to the terminal/command prompt and entering


`cd icdcoder`


***(4.5 optional step)*** *Create a virtual environment to work in. This step is preferable to not  
mess up other projects by installing the versions of the packages used in this project*

`python3 -m venv venv`

`source venv/bin/activate`


**(5)** Make sure you have the right packages installed as specified in the requirements file by entering

`pip install --upgrade pip`

`pip install -r requirements.txt`


**(6)** You are ready to train and/or evaluate your models. Follow the instructions in the sections below to do so


## Fine-tune/evaluate BERT model

Now, you can **(i)** fine-tune the pre-trained BERT model, **(ii)**, fine-tune the BERT pre-trained model and evaluate the newly fine-tuned model, or **(iii)** use and already fine-tuned model to predict the ICD codes of an unseen discharge summary. This is done by entering 

`python3 BERT_coder.py` followed by one of the main arguments specified in the section "Fine-tuning and testing using the main arguments" below.

### Fine-tuning and testing using the main arguments
*Note that the arguments are mutually exclusive and one is required to run the BERT_coder.py script*


#### Fine-tuning only

**(i)** `-train` 
                        Filepath to csv file used for training. The file
                        needs to follow the structure specified in the README section How to prepare dataset used for fine-tuning.

Train (fine-tune) using all of your dataset by using the entering


`python3 BERT_coder.py -train [filepath to data]`

For example

`python3 BERT_coder.py -train /Volumes/SecretUSB/train_data.csv`


#### Fine-tuning and evaluating

**(ii)** `-train_and_test`
			Filepath to csv file used for training and
                        testing. The file needs to follow the structure
                        specified in the README section How to prepare dataset used for fine-tuning.

Train and evaluate by entering


`python3 BERT_coder.py -train_and_test [filepath to data]`

For example

`python3 BERT_coder.py -train /Volumes/SecretUSB/train_and_test_data.csv`

#### Testing only

**(iii)** `-test`    	Use argument if you want to predict the ICD
                	codes of an unseen discharge summary

Predict the ICD codes of a single discharge summary already trained (fine-tuned) model by entering


`python3 BERT_coder.py -test`

Nothing more than the argument itself is specified. After entering the line above, you will be asked to enter the discharge summary you want to predict the ICD codes for.


### Customize run using the optional arguments

*To get a description of all arguments in the terminal/command prompt, use the help argument by entering the following*

`python3 BERT_coder.py -h`

#### For -train, -train_and_test, or -test

  `-pre_trained` 
                        Filepath to pre-trained model. Default is
                        subfolder ***./models/pre_trained_model***


  `-fine_tuned` 
                        Filepath to fine-tuned (traind) model. Default
                        is ***./models/fine_tuned_model/pytorch_model.bin***


  `-threshold` 
			                  The threshold that binarizes the model output
                        (0: label not present, 1: label present). Should be a
                        number between 0 and 1, **default is 0.5**.



#### For -train or -train_and_test


  `-new_fine_tuned` 
                        Filepath to save new fine-tuned model in


  `-batch_size_train`
                        The batch size for training. **Default is 4**.


  `-epochs`      	The number of epochs to train for. **Default is 10**.


  `-gradient_accumulation`
                        The gradient accumulation. The batch size
                        multiplied with the gradient accumulation is the
                        actual batch size. **Default is 8**.


  `-learning_rate`
                        The learning rate. **Default is 2e-5**.


  `-warm_up`      	The number of warm-up steps, that is, the number
                        of steps before the learning rate starts to decay.
                        **Default is 155**.


#### For -test or -train_and_test


  `-batch_size_train`
                        The batch size for testing. **Default is 2**.


#### For -train_and_test



  `-testsize`		
			Fraction of data to use for testing. Must be
                        between 0 and 1. **Default is 0.1**.


  `-kfold`        	
			The number of folds (k) to use in k-fold cross-
                        validation, must be > 1 for kfold to be used and
                        **default is 5**.


  `-random_state`
                        A seed (integer) to use as the random state in the
                        k-fold cross-validation. **Default is None**.


## Train/evaluate traditional supervised machine learning models

You are also ready to train and evaluate traditional supervised machine learning models...