from model import BasicModel
from data_load import DebatesDataset, create_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import time
import pickle
import optuna


def plot_graphs(train_loss, train_acc, test_loss, test_acc, save_folder):
    """
    Saves 2 graphs, One of train and test accuracy over epochs, and one of train and test loss over epochs.
    """
    plt.figure()
    plt.plot(train_acc, c="red", label='Train')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(test_acc, c="blue", label='Test')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'accuracy.png'))

    plt.figure()
    plt.plot(train_loss, c="red", label='Train')
    plt.plot(test_loss, c="blue", label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Loss over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'loss.png'))


def validate(loader: DataLoader, model, loss_function, device):
    """
    Calculate loss, average accuracy, average precision and average recall of a trained model on a dataset
    :param loader: Dataloader, data contains the true label for each sentence
    :param model: model that gives a score for each sentence and each label
    :return: loss, average accuracy, average precision and average recall
    Average accuracy means average percentage of correct sentence labeling across the debates.
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        accuracy = 0
        avg_precision = 0
        avg_recall = 0
        for batch_idx, input_data in enumerate(loader):
            sentence_embedding_tensor, tag_idx_tensor, debate_length = input_data
            scores = model(sentence_embedding_tensor)
            total_loss += loss_function(scores, tag_idx_tensor.view(-1).to(device))
            _, indices = torch.max(scores.detach().cpu(), 1)
            accuracy += torch.mean((tag_idx_tensor.clone().detach().cpu() == indices.clone().detach().cpu())
                                   .clone().detach(), dtype=torch.float)
            correct_speaker_changed_detection = (tag_idx_tensor.detach().cpu() * indices.detach().cpu()).sum()
            avg_precision += correct_speaker_changed_detection / indices.detach().cpu().sum()
            avg_recall += correct_speaker_changed_detection / tag_idx_tensor.detach().cpu().sum()
        avg_precision = avg_precision / len(loader)
        avg_recall = avg_recall / len(loader)
        accuracy = accuracy / len(loader)

    return total_loss / len(loader), accuracy * 100, avg_precision, avg_recall


def train(args, train_dataloader, test_dataloader, validation=True, loss_function=nn.MultiMarginLoss()):
    """
    Trains a basic model, and calculate best test accuracy.
    If validation is False: plots accuracies and losses,
        and saves the parameters of the model in the epoch with the best results.
    :param args: dictionary, hyper-parameters and other arguments
    :param train_dataloader: dataloader of the train dataset
    :param test_dataloader: dataloader of the test dataset
    :param validation: boolean, True if we only want to measure accuracy without plotting
    :param loss_function: loss function is defined by default as the MultiMarginLoss
    :return: max test accuracy among the epochs
    """
    # create a basic model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device}')
    tag_vocab_size = args['tag_vocab_size']
    model = BasicModel(
        embedding_dim=args['embedding_dim'],
        tag_vocab_size=tag_vocab_size,
        lstm_hidden_dimension=args['lstm_hidden_dimension'],
        lstm_n_layers=args['lstm_n_layers'],
        lstm_dropout= args['lstm_dropout'],
        unknown_token='[UNK]',
        dropout_alpha=args['dropout_alpha'])
    save_name = 'basic_model'

    # define an optimizer and a scheduler
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), lr=args['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(
                                                         train_dataloader) * args['epochs'])

    model.to(device)

    # Training start
    print("Training Started")

    train_accuracy_list = []
    train_loss_list = []

    test_accuracy_list = []
    test_loss_list = []

    best_accuracy = 0

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

        # Validate on test set
        test_loss, test_acc, test_precision, test_recall = validate(test_dataloader, model, loss_function, device)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        print(f"Epoch {epoch} Completed,\tLoss {train_loss:.3f}\t Train Accuracy: {train_acc:.3f}\t Test Accuracy: {test_acc:.3f}")

        torch.save(model.state_dict(), os.path.join(args['save_folder'], save_name + f'-{epoch}-{args["trial_number"]}.pth'))
    end = time.time()

    # Find best epoch and its accuracy
    best_model_index = np.argmax(test_accuracy_list)
    print(f'Best model index: {best_model_index + 1} with test accuracy: {best_accuracy}')
    if validation:
        return test_accuracy_list[best_model_index]

    # if validation is False, plot graphs, save best parameters and check again accuracy
    print(f'Total train time = {(end - start) / 60:.3f} min')
    plot_graphs(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, args['save_folder'])
    copyfile(os.path.join(args['save_folder'], save_name + f'-{best_model_index + 1}-{args["trial_number"]}.pth'),
             os.path.join(args['save_folder'], save_name + f'-{args["trial_number"]}.pthbest'))
    model.load_state_dict(torch.load(os.path.join(args['save_folder'], save_name +
                                                  f'-{args["trial_number"]}.pthbest')))

    # Just to check we get expected accuracy
    t0_test = time.time()
    _, test_acc, test_precision, test_recall = validate(test_dataloader, model, loss_function, device)
    t1_test = time.time()
    print(f'Best model test accuracy: {test_acc:.3f}, validation on test set took: {(t1_test - t0_test) / 60:.3f}')
    print('Done!')

    return test_acc


def split_train_validation(fold_num, k_fold, dataset):
    """
    Split dataset to train-set and test-set during fold_num-th iteration of cross-validation
    :param fold_num: number of iteration in cross-validation
    :param k_fold: number of folds
    :param dataset: dataset
    :return: train_set and test_set
    """
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)
    i = fold_num
    trll = 0
    trlr = i * seg
    vall = trlr
    valr = i * seg + seg
    trrl = valr
    trrr = total_size

    train_left_indices = list(range(trll, trlr))
    train_right_indices = list(range(trrl, trrr))
    train_indices = train_left_indices + train_right_indices
    val_indices = list(range(vall, valr))

    train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
    val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
    return train_set, val_set


def cross_validation(k_fold, dataset, args):
    """
    cross validation. Calculate mean accuracy of the validation sets
    :param k_fold: number of folds
    :param dataset: dataset
    :param args: arguments for training
    :return: mean accuracy of the validation sets
    """
    val_acc = 0
    for i in range(k_fold):
        print("Fold number is: ", i + 1)
        # split dataset to k_fold parts
        train_set, val_set = split_train_validation(i, k_fold,dataset)

        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set,shuffle=False)

        # Train a model using the train set and test it using the validation set
        val_acc += train(args, train_loader, val_loader)

    return val_acc/k_fold

def write_to_report(report_path, msg):
    with open(report_path, 'a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    path = 'debates'

    # create datasets. if datasets already exists, load them using pickle/load
    train_dataset, test_dataset = create_datasets(path, 'speaker changed only')
    """
    # Use this code to load pickle files of speaker-changed-only datasets
    with open('train_speaker_changed.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('test_speaker_changed.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    """

    # # Use this code to load pickle files of party-only datasets
    # with open('train_party.pkl', 'rb') as f:
    #     train_dataset = pickle.load(f)
    # with open('test_party.pkl', 'rb') as f:
    #     test_dataset = pickle.load(f)
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    def val_objective(trial):
        """
        Creates model with suggested (random) hyper-parameters,
        and computes cross-validation mean accuracy for those hyper-parameters.
        :return: cross-validation mean accuracy
        """
        args = {}
        args['epochs'] = 10
        args['lstm_hidden_dimension'] = trial.suggest_int('lstm_hidden_dimension', 50, 500)
        args['lstm_n_layers'] = trial.suggest_int('lstm_n_layers',2,3)
        args['lstm_dropout'] = trial.suggest_uniform('lstm_dropout', 0.01, 0.3)
        args['dropout_alpha'] = trial.suggest_uniform('dropout_alpha', 0.0, 0.3)
        args['lr'] = trial.suggest_uniform('lr', 0.001, 0.005)
        args['save_folder'] = f'./param_savings'
        args['trial_number'] = trial.number
        if not os.path.exists(args['save_folder']):
            os.mkdir(args['save_folder'])
        args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
        args['embedding_dim'] = train_dataset.sentence_vector_dim

        val_acc = cross_validation(5, train_dataset, args)
        return val_acc

    # choose best hyper-parameters
    study = optuna.create_study(direction="maximize")
    study.optimize(val_objective, timeout=60*60*0.01, n_jobs=3)
    print(f'Best params are: {study.best_params}\nBest accuracy is: {study.best_trial.value}')
    report_file = 'report_best_model.txt'
    if not os.path.exists('./output_files/'):
        os.mkdir('./output_files/')
    report_path = os.path.join('./output_files/', report_file)
    write_to_report(report_path,f'Best estimated cross validation accuracy is: {study.best_trial.value}')
    write_to_report(report_path,f'Best estimated cross validation hyper - parameters are: {study.best_trial.params}')

    # train model on all train set with chosen hyper-parameters and report accuracy
    best_args = study.best_trial.params
    best_args['epochs'] = 30
    best_args['save_folder'] = f'./party_best_param_savings'
    if not os.path.exists(best_args['save_folder']):
        os.mkdir(best_args['save_folder'])
    best_args['trial_number'] = ''
    best_args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
    best_args['embedding_dim'] = train_dataset.sentence_vector_dim
    test_acc = train(best_args,train_dataloader,test_dataloader,False)
    print(f"test acc is : {test_acc}")
    write_to_report(report_path,f'Test accuracy is: {test_acc}')

    # # Train and test without cross-validation
    # # You may use this code to train and test a model
    # args = dict()
    # args['epochs'] = 35
    # args['lstm_hidden_dimension'] = 100
    # args['lstm_n_layers'] = 2
    # args['lstm_dropout'] = 0.2
    # args['dropout_alpha'] = 0.
    # args['lr'] = 0.007
    # args['save_folder'] = f'./best_savings_with_new_training'
    # args['trial_number'] = ''
    # if not os.path.exists(args['save_folder']):
    #     os.mkdir(args['save_folder'])
    # args['tag_vocab_size'] = len(train_dataset.tag_idx_mappings)
    # args['embedding_dim'] = train_dataset.sentence_vector_dim
    # test_acc = train(args,train_dataloader,test_dataloader,False)
    # print("test_acc is : ", test_acc)
