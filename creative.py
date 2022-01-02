from model import BasicModel
from data_load import DebatesDataset, create_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
from shutil import copyfile
import time
import pickle
import optuna
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from train import plot_graphs, write_to_report, validate, split_train_validation


class F1_Loss(nn.Module):
    """
    Calculate a differentiable version of macro F1 score
    The original implementation is written by Michal Haltuf on Kaggle.
    :return: 1-F1 in order to be used for F1 maximization. torch.Tensor,  ndim` == 1.
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        # input is log_soft_max, hence exp(input) is a probability distribution
        y_pred = torch.exp(y_pred)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class speaker_changed_F1_Loss(nn.Module):
    """
    Calculate a differentiable version of F1 score of speaker-changed class
    :return: 1-F1 in order to be used for F1 maximization. torch.Tensor,  ndim` == 1.
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_true.ndim == 1
        y_pred = y_pred[:,1]  # look only at speaker-changed probabilities
        y_pred = torch.exp(y_pred)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1


def calculate_F1(precision, recall):
    """
    Calculates F1 score.
    Return 0 instead of NaN values.
    """
    with torch.no_grad():
        if precision + recall == 0 or math.isnan(precision) or math.isnan(recall):
            return 0
        return 2 * (precision * recall) / (precision + recall)


def plot_F1_graph(train_F1, test_F1, save_folder):
    """
    Saves a graph of train and test F1 measure over epochs.
    """
    plt.figure()
    plt.plot(train_F1, c="red", label='Train')
    plt.xlabel("Epochs")
    plt.ylabel("F1")

    plt.plot(test_F1, c="blue", label='Test')
    plt.title('F1 over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'F1.png'))


def train(args, train_dataloader, test_dataloader, validation=True, loss_function=F1_Loss()):
    """
    Trains a creative model, and calculate best test F1 measre.
    If validation is False: plots accuracies, F1s and losses,
        and saves the parameters of the model in the epoch with the best results.
    :param args: dictionary, hyper-parameters and other arguments
    :param train_dataloader: dataloader of the train dataset
    :param test_dataloader: dataloader of the test dataset
    :param validation: boolean, True if we only want to measure F1 without pots
    :param loss_function: loss function is defined by default as F1_Loss
    :return: max test F1 among the epochs
    """
    # create a basic model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device}')
    model = BasicModel(
        embedding_dim=args['embedding_dim'],
        tag_vocab_size= args['tag_vocab_size'],
        lstm_hidden_dimension=args['lstm_hidden_dimension'],
        lstm_n_layers=args['lstm_n_layers'],
        lstm_dropout= args['lstm_dropout'],
        unknown_token='[UNK]',
        dropout_alpha=args['dropout_alpha'])
    save_name = 'creative_model'

    # define an optimizer and a scheduler
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), lr=args['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(
                                                         train_dataloader) * args['epochs'])

    model.to(device)

    # Training start
    print("Training Started")

    train_accuracy_list = []
    train_F1_list = []
    train_loss_list = []

    test_accuracy_list = []
    test_F1_list = []
    test_loss_list = []

    best_F1 = 0

    epochs = args['epochs']
    start = time.time()
    for epoch in range(1, epochs + 1, 1):
        train_loss = 0  # To keep track of the loss value
        i = 0

        # Training. Batch size is always 1.
        model.train()
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            sentence_embedding_tensor, tag_idx_tensor, debate_length = input_data
            scores = model(sentence_embedding_tensor)
            loss = loss_function(scores, tag_idx_tensor.view(-1).to(device))
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()
            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(float(train_loss))

        # Validate on train set
        _, train_acc, train_precision, train_recall = validate(train_dataloader, model, loss_function, device)
        train_accuracy_list.append(float(train_acc))
        train_F1 = calculate_F1(train_precision, train_recall)
        train_F1_list.append(train_F1)

        # Validate on test set
        test_loss, test_acc, test_precision, test_recall = validate(test_dataloader, model, loss_function, device)
        test_accuracy_list.append(test_acc)
        test_F1 = calculate_F1(test_precision, test_recall)
        test_F1_list.append(test_F1)
        test_loss_list.append(test_loss)
        if test_F1 > best_F1:
            best_F1 = test_F1
        print(f"Epoch {epoch} Completed,\tLoss {train_loss:.3f}\t Train F1: {train_F1:.3f}\t Test F1: {test_F1:.3f}")

        torch.save(model.state_dict(), os.path.join(args['save_folder'], save_name + f'-{epoch}-{args["trial_number"]}.pth'))
    end = time.time()

    # Find best epoch and its F1 measure
    best_model_index = np.argmax(test_F1_list)
    print(f'Best model index: {best_model_index + 1} with test F1: {best_F1}')
    if validation:
        return test_F1_list[best_model_index]

    # if validation is False, plot graphs, save best parameters and check again F1
    print(f'Total train time = {(end - start) / 60:.3f} min')
    plot_graphs(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, args['save_folder'])
    plot_F1_graph(train_F1_list, test_F1_list, args['save_folder'])
    copyfile(os.path.join(args['save_folder'], save_name + f'-{best_model_index + 1}-{args["trial_number"]}.pth'),
             os.path.join(args['save_folder'], save_name + f'-{args["trial_number"]}.pthbest'))
    model.load_state_dict(torch.load(os.path.join(args['save_folder'], save_name +
                                                  f'-{args["trial_number"]}.pthbest')))

    # Just to check we get expected accuracy
    t0_test = time.time()
    _, test_acc, test_precision, test_recall = validate(test_dataloader, model, loss_function, device)
    test_F1 = calculate_F1(test_precision, test_recall)
    t1_test = time.time()
    print(f'Best model test accuracy: {test_acc:.3f}, ' +
          f'Best model test F1: {test_F1:.3f}, ' +
          f'validation on test set took: {(t1_test - t0_test) / 60:.3f}')
    print(f'Best model test precision: {test_precision:.3f}, ' +
          f'Best model recall: {test_recall:.3f}')
    print('Done!')

    return test_F1


def cross_validation(k_fold, dataset, args):
    """
    cross validation. Calculate mean accuracy of the validation sets
    :param k_fold: number of folds
    :param dataset: dataset
    :param args: arguments for training
    :return: mean output score (F1) of the validation sets
    """
    val_measure = 0
    for i in range(k_fold):
        # split dataset to k_fold parts
        train_set, val_set = split_train_validation(i, k_fold,dataset)

        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set,shuffle=False)

        # Train a model using the train set and test it using the validation set
        val_measure += train(args, train_loader, val_loader)

    return val_measure/k_fold


if __name__ == '__main__':
    path = 'debates'

    # create datasets. if datasets already exists, load them using pickle/load
    train_dataset, test_dataset = create_datasets(path, 'speaker changed only')
    # Use this code to load pickle files of party-only datasets
    # with open('train_speaker_changed.pkl', 'rb') as f:
    #     train_dataset = pickle.load(f)
    # with open('test_speaker_changed.pkl', 'rb') as f:
    #     test_dataset = pickle.load(f)
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    # def val_objective(trial):
    #     """
    #     Creates model with suggested (random) hyper-parameters,
    #     and computes cross-validation mean accuracy for those hyper-parameters.
    #     :return: cross-validation mean accuracy
    #     """
    #     args = {}
    #     args['epochs'] = 1
    #     args['lstm_hidden_dimension'] = trial.suggest_int('lstm_hidden_dimension', 50, 400)
    #     args['lstm_n_layers'] = trial.suggest_int('lstm_n_layers',2,3)
    #     args['lstm_dropout'] = trial.suggest_uniform('lstm_dropout', 0.01, 0.6)
    #     args['dropout_alpha'] = trial.suggest_uniform('dropout_alpha', 0.01, 0.6)
    #     args['lr'] = trial.suggest_uniform('lr', 0.001, 0.05)
    #     args['save_folder'] = f'./param_savings'
    #     args['trial_number'] = trial.number
    #     if not os.path.exists(args['save_folder']):
    #         os.mkdir(args['save_folder'])
    #     args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
    #     args['embedding_dim'] = train_dataset.sentence_vector_dim
    #
    #     val_acc = cross_validation(5, train_dataset, args)
    #     return val_acc
    #
    # # choose best hyper-parameters
    # study = optuna.create_study(direction="maximize")
    # study.optimize(val_objective, timeout=60*60*0.01, n_jobs=3)
    # print(f'Best params are: {study.best_params}\nBest F1 is: {study.best_trial.value}')
    # report_file = 'report_best_model.txt'
    # if not os.path.exists('./output_files/'):
    #     os.mkdir('./output_files/')
    # report_path = os.path.join('./output_files/', report_file)
    # write_to_report(report_path,f'Best estimated cross validation F1 is: {study.best_trial.value}')
    # write_to_report(report_path,f'Best estimated cross validation hyper - parameters are: {study.best_trial.params}')
    #
    # # train model on all train set with chosen hyper-parameters and report F1
    # best_args = study.best_trial.params
    # best_args['epochs'] = 10
    # best_args['save_folder'] = f'./party_best_param_savings'
    # if not os.path.exists(best_args['save_folder']):
    #     os.mkdir(best_args['save_folder'])
    # best_args['trial_number'] = ''
    # best_args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
    # best_args['embedding_dim'] = train_dataset.sentence_vector_dim
    #
    # test_F1 = train(best_args, train_dataloader, test_dataloader, False, loss_function=speaker_changed_F1_Loss())
    # print(f"test F1 is : {test_F1}")
    # write_to_report(report_path,f'Test F1 is: {test_F1}')

    # train and test without cross-validation
    # You may use this code to train and test a model
    args = dict()
    args['epochs'] = 30
    args['lstm_hidden_dimension'] = 80
    args['lstm_n_layers'] = 2
    args['lstm_dropout'] = 0.2
    args['dropout_alpha'] = 0.11
    args['lr'] = 0.008
    args['save_folder'] = f'./best_savings_with_new_training'
    args['trial_number'] = ''
    if not os.path.exists(args['save_folder']):
        os.mkdir(args['save_folder'])
    args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
    args['embedding_dim'] = train_dataset.sentence_vector_dim

    # results of model trained with F1_loss
    test_F1 = train(args, train_dataloader, test_dataloader, False)
    print("test F1 is : ", test_F1)

    # compare results to results of model trained with speaker_changed_F1_Loss
    test_F1_optimizing_speaker_chanegd_F1 = train(args, train_dataloader, test_dataloader,
                                        False, loss_function=speaker_changed_F1_Loss())
    print("test F1 in optimizing_speaker_chanegd_F1 training is : ", test_F1_optimizing_speaker_chanegd_F1)

    # compare results to results of trained basic model
    test_F1_optimizing_accuracy = train(args, train_dataloader, test_dataloader,
                                        False, loss_function=nn.MultiMarginLoss())
    print("test F1 in basic model training is : ", test_F1_optimizing_accuracy)