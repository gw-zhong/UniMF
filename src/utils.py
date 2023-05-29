import torch
import os
import random
import numpy as np
from src.dataset import Multimodal_Datasets
from torch.utils.data import TensorDataset


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def load_meld(mode, path):
    mode = mode.lower()
    text_dim = 600
    names_data = ['train_x_{}.pt', 'val_x_{}.pt', 'test_x_{}.pt',
                  'train_y_{}.pt', 'val_y_{}.pt', 'test_y_{}.pt',
                  'train_mask_{}.pt', 'val_mask_{}.pt', 'test_mask_{}.pt']

    data_path = path
    x_train = torch.load(data_path + names_data[0].format(mode)).to(dtype=torch.float32)
    x_valid = torch.load(data_path + names_data[1].format(mode)).to(dtype=torch.float32)
    x_test = torch.load(data_path + names_data[2].format(mode)).to(dtype=torch.float32)
    y_train = torch.load(data_path + names_data[3].format(mode)).to(dtype=torch.long)
    y_valid = torch.load(data_path + names_data[4].format(mode)).to(dtype=torch.long)
    y_test = torch.load(data_path + names_data[5].format(mode)).to(dtype=torch.long)
    mask_train = torch.load(data_path + names_data[6].format(mode)).to(dtype=torch.float32)
    mask_valid = torch.load(data_path + names_data[7].format(mode)).to(dtype=torch.float32)
    mask_test = torch.load(data_path + names_data[8].format(mode)).to(dtype=torch.float32)

    classes = torch.max(y_train).item() + 1
    total_dim = x_train.size(2)
    train_set = TensorDataset(x_train[:, :, text_dim:], x_train[:, :, :text_dim], mask_train, y_train)
    valid_set = TensorDataset(x_valid[:, :, text_dim:], x_valid[:, :, :text_dim:], mask_valid, y_valid)
    test_set = TensorDataset(x_test[:, :, text_dim:], x_test[:, :, :text_dim:], mask_test, y_test)

    return classes, {'audio_dim': total_dim - text_dim, 'text_dim': text_dim,
                     'n_train': x_train.size(0), 'n_valid': x_valid.size(0), 'n_test': x_test.size(0),
                     'num_utterance': x_train.size(
                         1)}, train_set, valid_set, test_set, mask_train, mask_valid, mask_test


def save_load_name(args, name=''):
    if args.aligned:
        tmpname = args.dataset + '_' + 'aligned'
    elif not args.aligned:
        tmpname = args.dataset + '_' + 'nonaligned'

    # return name + '_' + args.modalities + '_' + args.model
    return tmpname + '_' + args.modalities + '_' + name


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')
    # print(f"Saved model at pre_trained_models/{name}.pt!")


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
