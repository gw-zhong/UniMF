# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Select GPU when not use distribute
import csv
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train, train_meld, train_urfunny, train_urfunny_bert
from humor_dataloader import HumorDataset, load_pickle
from bert_dataloader import MMDataset
from humor_bert_dataloader import HumorBertDataset

import optuna
from optuna.samplers import TPESampler

parser = argparse.ArgumentParser(description='Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='UniMF',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--aligned', type=bool, default=True,
                    help='consider aligned experiment or not (default: True)')
parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'urfunny', 'mosi-bert', 'mosei-bert', 'meld_senti', 'meld_emo'],
                    help='dataset to use (default: mosi)')
parser.add_argument('--data_path', type=str, default='/root/autodl-tmp',
                    help='path for storing the dataset')
parser.add_argument('--run_id', type=int, default=1,
                    help='experiment id (default: 1)')
parser.add_argument('--trials', type=int, default=10,
                    help='number of trials (default: 10)')

# Dropouts
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')

# Architecture
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network (default: 8)')

# Tuning
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs (default: 100)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='do not use cuda (default: false)')
parser.add_argument('--distribute', action='store_true',
                    help='use distributed training (default: false)')
parser.add_argument('--name', type=str, default='UniMF',
                    help='name of the trial (default: "UniMF")')
parser.add_argument('--use_bert', action='store_true', help='whether to use bert embeddings')
parser.add_argument('--language', type=str, default='en',
                    help='bert language (default: en)')

# Input modality settings
parser.add_argument('--modalities', type=str, default='LAV', choices=['LAV', 'LA', 'LV', 'AV', 'L', 'A', 'V'],
                    help='input modalities (default: LAV)')

args = parser.parse_args()

seed_everything(args)
dataset = str.lower(args.dataset.strip())

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8,
    'meld_senti': 3,
    'meld_emo': 7,
    'urfunny': 1,
    'mosi-bert': 1,
    'mosei-bert': 1,
    'sims': 1
}

criterion_dict = {
    'mosi': 'L1Loss',
    'mosei_senti': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'meld_senti': 'CrossEntropyLoss',
    'meld_emo': 'CrossEntropyLoss',
    'urfunny': 'BCEWithLogitsLoss',
    'mosi-bert': 'L1Loss',
    'mosei-bert': 'L1Loss',
    'sims': 'L1Loss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        seed_everything(args)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

if dataset == 'mosi' or dataset == 'mosei_senti' or dataset == 'iemocap':
    if args.use_bert:
        raise ValueError('If you want to run bert version, please use mosi-bert or mosei-bert')
    train_data = get_data(args, dataset, 'train')
    valid_data = get_data(args, dataset, 'valid')
    test_data = get_data(args, dataset, 'test')
elif dataset == 'meld_senti':
    classes, meld_info, train_data, valid_data, test_data, mask_train_data, mask_valid_data, mask_test_data \
        = load_meld('sentiment', os.path.join(args.data_path, 'MELD/'))
elif dataset == 'meld_emo':
    classes, meld_info, train_data, valid_data, test_data, mask_train_data, mask_valid_data, mask_test_data \
        = load_meld('emotion', os.path.join(args.data_path, 'MELD/'))
elif dataset == 'urfunny':
    path = './data/UR-FUNNY/'
    max_context_len = 8
    max_sen_len = 40
    if not args.use_bert:
        train_data = HumorDataset('train', path, max_context_len, max_sen_len, online=False)
        valid_data = HumorDataset('dev', path, max_context_len, max_sen_len, online=False)
        test_data = HumorDataset('test', path, max_context_len, max_sen_len, online=False)
    else:
        args.weight_decay_bert = 0.001
        args.lr_bert = 5e-5
        train_data = HumorBertDataset('train', path, max_context_len, max_sen_len, online=False)
        valid_data = HumorBertDataset('dev', path, max_context_len, max_sen_len, online=False)
        test_data = HumorBertDataset('test', path, max_context_len, max_sen_len, online=False)
elif dataset == 'mosi-bert' or dataset == 'mosei-bert' or dataset == 'sims':
    args.use_bert = True
    args.language = 'cn' if dataset == 'sims' else 'en'
    args.weight_decay_bert = 0.001
    args.lr_bert = 5e-5
    train_data = MMDataset(args, 'train')
    valid_data = MMDataset(args, 'valid')
    test_data = MMDataset(args, 'test')
else:
    raise ValueError('Unknown dataset')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

print('Finish loading the data....')
print(f'### Dataset - {str.upper(dataset)}')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args

# Fix feature dimensions and sequence length
if dataset == 'mosi' or dataset == 'mosei_senti' or dataset == 'iemocap':
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
elif dataset == 'meld_senti':
    hyp_params.orig_d_l, hyp_params.orig_d_a = 600, 600
    hyp_params.l_len, hyp_params.a_len = 33, 33
elif dataset == 'meld_emo':
    hyp_params.orig_d_l, hyp_params.orig_d_a = 600, 300
    hyp_params.l_len, hyp_params.a_len = 33, 33
elif dataset == 'urfunny':
    if not hyp_params.use_bert:
        # hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 300, 81, 75  # for UR-FUNNY V1
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 300, 81, 371  # for UR-FUNNY V2
        # hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 360, 360, 360  # context (8*40) + target(40)
        hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 40, 40, 40  # only target
    else:
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 81, 371  # for UR-FUNNY V2
        hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 40, 40, 40  # only target
elif dataset == 'mosi-bert' or dataset == 'mosei-bert':
    if dataset == 'mosi-bert':
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 5, 20
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 375, 500
    else:
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 74, 35
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 500, 500
elif dataset == 'sims':
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 33, 709
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 39, 400, 55
else:
    raise ValueError('Unknown dataset')

hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
hyp_params.embed_dim = 32

if 'meld' in dataset and hyp_params.modalities == 'LAV':
    hyp_params.modalities = 'LA'
print(f'### Note: You are running in input modality(es) = {hyp_params.modalities} mode.')
if hyp_params.use_bert:
    print(f'### Note: You are running in BERT mode.')

if __name__ == '__main__':
    sampler = TPESampler(seed=args.seed)

    def objective(trial):
        if dataset != 'meld_senti' and dataset != 'meld_emo' and hyp_params.modalities != 'LAV' or \
                'meld' in dataset and hyp_params.modalities != 'LA':
            hyp_params.trans_layers = trial.suggest_int('trans_layers', 1, 8, step=1)
            hyp_params.trans_dropout = trial.suggest_uniform('trans_dropout', 0.0, 0.5)
        hyp_params.attn_dropout = trial.suggest_uniform('attn_dropout', 0.0, 0.5)
        hyp_params.embed_dropout = trial.suggest_uniform('embed_dropout', 0.0, 0.5)
        hyp_params.out_dropout = trial.suggest_uniform('out_dropout', 0.0, 0.5)
        hyp_params.l_kernel_size = trial.suggest_int('l_kernel_size', 1, 5, step=2)
        if dataset != 'meld_senti' and dataset != 'meld_emo':
            hyp_params.v_kernel_size = trial.suggest_int('v_kernel_size', 1, 5, step=2)
        hyp_params.a_kernel_size = trial.suggest_int('a_kernel_size', 1, 5, step=2)
        hyp_params.multimodal_layers = trial.suggest_int('multimodal_layers', 1, 8, step=1)

        if dataset == 'mosi' or dataset == 'mosei_senti' or dataset == 'iemocap' or \
                dataset == 'mosi-bert' or dataset == 'mosei-bert' or dataset == 'sims':
            acc = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
        elif dataset == 'meld_senti' or dataset == 'meld_emo':
            acc = train_meld.initiate(hyp_params, train_loader, valid_loader, test_loader)
        elif dataset == 'urfunny':
            if not hyp_params.use_bert:
                acc = train_urfunny.initiate(hyp_params, train_loader, valid_loader, test_loader, onlypunch=True)
            else:
                acc = train_urfunny_bert.initiate(hyp_params, train_loader, valid_loader, test_loader)
        else:
            raise ValueError('Unknown dataset')

        return acc

    n_trials = hyp_params.trials
    if dataset == 'iemocap':
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize', 'maximize'], sampler=sampler,
            study_name='exp_test_' + str(hyp_params.run_id))
        study.optimize(objective, n_trials=n_trials)
        print('-' * 50)
        emos = ["Neutral", "Happy", "Sad", "Angry"]
        for emo_ind in range(4):
            best_trial = max(study.best_trials, key=lambda t: t.values[emo_ind])
            print(f"{emos[emo_ind]}: ")
            print(f'Best Accuracy is: {best_trial.values[emo_ind]:.4f}')
            print(f'Best Hyperparameters are:\n{best_trial.params}')
    else:
        study = optuna.create_study(
            direction='maximize', sampler=sampler, study_name='exp_test_' + str(hyp_params.run_id))
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        print('-' * 50)
        print(f'Best Accuracy is: {best_trial.value:.4f}')
        print(f'Best Hyperparameters are:\n{best_trial.params}')

    if hyp_params.aligned:
        df = study.trials_dataframe().to_csv(
            'results/{}_aligned_{}_trials_{}.csv'.format(dataset, hyp_params.modalities, n_trials))
    else:
        df = study.trials_dataframe().to_csv(
            'results/{}_unaligned_{}_trials_{}.csv'.format(dataset, hyp_params.modalities, n_trials))
