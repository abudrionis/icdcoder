'''
Script with collection of funcitons used for finetuning and evaluating 

Author: Anastasios Lamproudis

'''

import torch
import transformers
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from sklearn.model_selection import KFold
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle as pkl
import os


class Trainer:
    def __init__(self, model, tokenizer):
        super(Trainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optim = None
        self.epoch_loss_history = []
        self.val_epoch_loss_history = []
        self.epoch_f1_history = []
        self.val_epoch_f1_history = []
        self.epoch_acc_history = []
        self.val_epoch_acc_history = []
        self.epoch_precision_history = []
        self.val_epoch_precision_history = []
        self.epoch_recall_history = []
        self.val_epoch_recall_history = []
        self.history = {}

    def train(self, X, Y, epochs: int, batch_size: int, learning_rate: float, gradient_accumulation: int = 1, X_val = None,
              Y_val = None, thres: float = .5, warm_up: int = 155, fold = 0, return_best_model = True, save_path = './'):
        if torch.cuda.is_available():
            dev = 'cuda:0'
            self.model.to(dev)
        else:
            dev = 'cpu'
        dataLoader = self.get_DataLoader(X = X, Y = Y, batch_size = batch_size)
        if X_val:
            val_dataLoader = self.get_DataLoader(X = X_val, Y = Y_val, batch_size = batch_size)
        if not self.optim:
            self.optim = self.get_optim(learning_rate = learning_rate)
        scheduler = self.get_scheduler(steps = (len(dataLoader) // gradient_accumulation) * epochs, warm_up = warm_up)
        old_loss, old_f1, old_acc = np.inf, -np.inf, -np.inf
        model_best_loss, model_best_f1, model_best_acc = None, None, None
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}')
            self.model.train()
            batch_loss_history, batch_f1_history, batch_acc_history = [], [], []
            batch_precision_history, batch_recall_history = [], []
            for b, (i, a, y) in enumerate(tqdm(dataLoader)):
                output = self.model(i.to(dev), a.to(dev))
                loss_function = BCEWithLogitsLoss()
                batch_loss = loss_function(output, y.to(dev).float())
                batch_loss_history.append(batch_loss.cpu().detach().item())
                batch_loss.backward()
                if b % (gradient_accumulation - 1) == 0 and b != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                    self.optim.step()
                    self.model.zero_grad()
                    scheduler.step()
                output = torch.sigmoid(input = output)
                output = output.cpu().detach().numpy()
                y = y.cpu().numpy()
                output = (output >= thres).astype(int)
                batch_f1_history.append(f1_score(y, output, average = 'micro'))
                batch_acc_history.append(self.accuracy(y, output))
                batch_recall_history.append(recall_score(y_true = y, y_pred = output, average = 'micro'))
                batch_precision_history.append(precision_score(y_true = y, y_pred = output, average = 'micro', zero_division=False))
            self.optim.step()
            self.model.zero_grad()
            self.epoch_f1_history.append(np.average(batch_f1_history))
            self.epoch_loss_history.append(np.average(batch_loss_history))
            self.epoch_acc_history.append(np.average(batch_acc_history))
            self.epoch_recall_history.append(np.average(batch_recall_history))
            self.epoch_precision_history.append(np.average(batch_precision_history))

            # Evaluation of the validation set if it exists
            if X_val:
                self.model.eval()
                with torch.no_grad():
                    batch_loss_history, batch_f1_history, batch_acc_history = [], [], []
                    batch_precision_history, batch_recall_history = [], []
                    dummy_predictions, dummy_labels = np.empty((1, len(Y[0]))), np.empty((1, len(Y[0])))
                    for i_val, a_val, y_val in val_dataLoader:
                        output = self.model(i_val.to(dev), a_val.to(dev))
                        loss_function = BCEWithLogitsLoss()
                        batch_loss = loss_function(output, y_val.to(dev).float())
                        batch_loss_history.append(batch_loss.cpu().detach().item())
                        output = torch.sigmoid(input=output)
                        output = output.cpu().detach().numpy()
                        y_val = y_val.cpu().numpy()
                        output = (output >= thres).astype(int)
                        batch_f1_history.append(f1_score(y_val, output, average = 'micro'))
                        batch_acc_history.append(self.accuracy(y_val, output))
                        batch_recall_history.append(recall_score(y_true = y_val, y_pred = output, average = 'micro'))
                        batch_precision_history.append(precision_score(y_true = y_val, y_pred = output, average = 'micro', zero_division=False))
                        dummy_predictions = np.concatenate((dummy_predictions, output), axis = 0)
                        dummy_labels = np.concatenate((dummy_labels, y_val), axis = 0)
                    predictions, true_labels = dummy_predictions[1:], dummy_labels[1:]
                    log_path = f'{save_path}/Fold__{fold + 1}/Epoch_{epoch + 1}_logs_fold.txt'
                    os.makedirs(os.path.dirname(log_path), exist_ok = True)
                    with open(log_path, 'w') as text_file:
                        text_file.write(classification_report(true_labels, predictions, zero_division=False))
                    if np.average(batch_loss_history) < old_loss:
                        loss_epoch = epoch + 1
                        model_best_loss = deepcopy(self.model).cpu()
                        old_loss = np.average(batch_loss_history)
                    self.val_epoch_f1_history.append(np.average(batch_f1_history))
                    self.val_epoch_loss_history.append(np.average(batch_loss_history))
                    self.val_epoch_acc_history.append(np.average(batch_acc_history))
                    self.val_epoch_recall_history.append(np.average(batch_recall_history))
                    self.val_epoch_precision_history.append(np.average(batch_precision_history))

                print('\n%-30s %4.5f' % ('Validation loss (BCE)', self.val_epoch_loss_history[-1]))
                print('%-30s %4.5f' % ('Train loss (BCE)', self.epoch_loss_history[-1]))
                print('%-30s %4.2f' % ('Validation F1-micro score', self.val_epoch_f1_history[-1]))
                print('%-30s %4.2f' % ('Train F1-micro score', self.epoch_f1_history[-1]))
            else:
                print('\n%-30s %4.5f' % ('Train loss (BCE)', self.epoch_loss_history[-1]))
                print('%-30s %4.2f' % ('Train F1-micro score', self.epoch_f1_history[-1]))

        if X_val:
            torch.save(model_best_loss, f = f'{save_path}/Fold__{fold + 1}/best_loss_model_epoch_{loss_epoch}.bin')
            self.history['validation loss'] = self.val_epoch_loss_history
            self.history['validation accuracy'] = self.val_epoch_acc_history
            self.history['validation f1 score'] = self.val_epoch_f1_history
            self.history['validation recall'] = self.val_epoch_recall_history
            self.history['validation precision'] = self.val_epoch_precision_history
        else:
            if not os.path.exists(f'{save_path}/Fold__{fold + 1}'):
                os.makedirs(f'{save_path}/Fold__{fold + 1}')
            torch.save(self.model.cpu(), f = f'{save_path}/Fold__{fold + 1}/model_epoch_{epochs}.bin')
        self.history['train loss'] = self.epoch_loss_history
        self.history['train accuracy'] = self.epoch_acc_history
        self.history['train f1 score'] = self.epoch_f1_history
        self.history['train recall'] = self.epoch_recall_history
        self.history['train precision'] = self.epoch_precision_history
        pkl.dump(self.history, open(f'{save_path}/Fold__{fold + 1}/history.pkl', 'wb'))
        if return_best_model:
            self.model = model_best_loss

    def get_DataLoader(self, X, Y, batch_size):
        X = self.tokenize(X = X)
        i, a = X['input_ids'], X['attention_mask']
        dataset = TensorDataset(torch.tensor(i), torch.tensor(a), torch.tensor(Y))
        sampler = RandomSampler(dataset)
        return DataLoader(dataset = dataset, sampler = sampler, batch_size = batch_size)

    def get_evalDataLoader(self, X, batch_size):
        X = self.tokenize(X = X)
        i, a = X['input_ids'], X['attention_mask']
        dataset = TensorDataset(torch.tensor(i), torch.tensor(a))
        return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)

    def tokenize(self, X):
        return self.tokenizer.batch_encode_plus(X, padding ='longest', truncation = True, return_token_type_ids = False)

    def get_optim(self, learning_rate):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': .01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': .0}
        ]
        return AdamW(params = optimizer_grouped_parameters, lr = learning_rate)

    def get_scheduler(self, steps, warm_up):
        return transformers.get_linear_schedule_with_warmup(
            optimizer = self.optim,
            num_warmup_steps = warm_up,
            num_training_steps = steps
        )

    def evaluate(self, X, num_labels, batch_size, thres = .5):
        if torch.cuda.is_available():
            dev = 'cuda:0'
            self.model.to(dev)
        else:
            dev = 'cpu'
        dataLoader = self.get_evalDataLoader(X = X, batch_size = batch_size)
        self.model.eval()
        with torch.no_grad():
            dummy = np.empty((1, num_labels))
            for i, a in dataLoader:
                output = self.model(i.to(dev), a.to(dev))
                output = torch.sigmoid(input = output)
                output = output.cpu().detach().numpy()
                output = (output >= thres).astype(int)
                dummy = np.concatenate((dummy, output), axis = 0)
        return dummy[1:]

    def plot(self, fold):
        epochs = len(self.epoch_loss_history)
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize = (19.2, 10.8), dpi = 100)
        ax1.plot(range(epochs), self.epoch_loss_history, label = 'Train')
        if len(self.val_epoch_loss_history) > 0:
            ax1.plot(range(epochs), self.val_epoch_loss_history, label = 'Validation')
        ax1.set_title('Loss'), ax1.legend(), ax1.grid(), ax1.set(ylabel = 'Loss')
        ax2.plot(range(epochs), self.epoch_acc_history, label = 'Train')
        if len(self.val_epoch_acc_history) > 0:
            ax2.plot(range(epochs), self.val_epoch_acc_history, label = 'Validation')
        ax2.set_title('Accuracy'), ax2.legend(), ax2.grid(), ax2.set(ylabel = 'Accuracy')
        ax3.plot(range(epochs), self.epoch_f1_history, label = 'Train')
        if len(self.val_epoch_f1_history) > 0:
            ax3.plot(range(epochs), self.val_epoch_f1_history, label = 'Validation')
        ax3.set_title('F1-micro score'), ax3.legend(), ax3.grid(), ax3.set(ylabel = 'F1-micro score')

        for ax in fig.get_axes():
            ax.label_outer()

        plt.xlabel('Epochs')
        fig.savefig(fname = f'./fold_{fold+1}.png')

    @staticmethod
    def accuracy(y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return np.mean(acc_list)


class KFoldCrossVal:
    def __init__(self, nfolds, trainer, RANDOM_STATE):
        self.nfolds = nfolds
        self.trainer = trainer
        self.random_state = np.random.seed(RANDOM_STATE)

    def train(self, X, Y, epochs = 10, batch_size = 4, learning_rate = 2e-5, gradient_accumulation = 8, thres = 0.5, warm_up = 155, save_path='./'):
        k_fold = KFold(n_splits = self.nfolds, random_state = self.random_state, shuffle = True)
        dummy_predictions = np.empty((1, len(Y[0])))
        dummy_labels = np.empty((1, len(Y[0])))

        for fold, (train_indices, test_indices) in enumerate(k_fold.split(X=X)):
            new_trainer = deepcopy(self.trainer)
            x, y = [X[i] for i in train_indices], [Y[i] for i in train_indices]
            x_test, y_test = [X[i] for i in test_indices], [Y[i] for i in test_indices]
            new_trainer.train(X = x, Y = y, X_val = x_test, Y_val = y_test,
                              epochs = epochs,
                              batch_size = batch_size,
                              learning_rate = learning_rate,
                              gradient_accumulation = gradient_accumulation,
                              thres = thres,
                              warm_up = warm_up,
                              fold = fold,
                              return_best_model = True,
                              save_path = save_path
                              )
            predictions = new_trainer.evaluate(X = x_test,
                                               num_labels = self.trainer.model.num_labels,
                                               batch_size = batch_size,
                                               thres = thres)
            log_path = f'{save_path}/Fold__{fold + 1}/final_epoch_logs.txt'
            with open(log_path, 'w') as text_file:
                text_file.write(classification_report(y_test, predictions, zero_division=False))
            dummy_predictions = np.concatenate((dummy_predictions, predictions), axis = 0)
            dummy_labels = np.concatenate((dummy_labels, np.array(y_test)), axis = 0)

        predictions = dummy_predictions[1:]
        true_labels = dummy_labels[1:]
        with open(f'{save_path}/final_combined_logs.txt', 'w') as text_file:
            text_file.write(classification_report(true_labels, predictions, zero_division=False))
        print('\nCombined results for all folds of the data\n')
        print(classification_report(true_labels, predictions, zero_division=False))

    def plot(self, loss, accuracy, f1):
        epochs = loss.shape[1]
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize = (19.2, 10.8), dpi = 100)
        loss_std, loss_mean = np.std(loss, axis = 0), np.mean(loss, axis = 0)
        ax1.plot(range(epochs), loss_mean)
        ax1.fill_between(range(epochs), loss_mean - loss_std, loss_mean + loss_std, alpha = .2)
        ax1.set_title('Validation loss (BCE)'), ax1.legend(), ax1.grid(), ax1.set(ylabel = 'Loss')
        acc_std, acc_mean = np.std(accuracy, axis = 0), np.mean(accuracy, axis = 0)
        ax2.plot(range(epochs), acc_mean)
        ax2.fill_between(range(epochs), acc_mean - acc_std, acc_mean + acc_std, alpha = .2)
        ax2.set_title('Validation accuracy'), ax2.legend(), ax2.grid(), ax2.set(ylabel = 'Accuracy')
        f1_std, f1_mean = np.std(f1, axis = 0), np.mean(f1, axis = 0)
        ax3.plot(range(epochs), f1_mean)
        ax3.fill_between(range(epochs), f1_mean - f1_std, f1_mean + f1_std, alpha = .2)
        ax3.set_title('Validation F1-micro score'), ax3.legend(), ax3.grid(), ax3.set(ylabel = 'F1-micro score')

        for ax in fig.get_axes():
            ax.label_outer()

        plt.xlabel('Epochs')
        fig.show()
        fig.savefig(fname = f'./{self.nfolds}_fold_metrics.png')
