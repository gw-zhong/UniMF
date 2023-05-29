import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'A':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        trans_criterion = getattr(nn, 'MSELoss')()
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
            settings = {'model': model,
                        'translator': translator,
                        'translator_optimizer': translator_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        else:
            raise ValueError('Unknown modalities type')
    elif hyp_params.modalities == 'LA':
        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    if hyp_params.modalities != 'LA':
        trans_criterion = settings['trans_criterion']
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']
        else:
            raise ValueError('Unknown modalities type')
    else:
        translator = None

    def train(model, translator, optimizer, criterion):
        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'LA':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                translator.train()
            else:
                raise ValueError('Unknown modalities type')
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):

            model.zero_grad()
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                    translator.zero_grad()
                else:
                    raise ValueError('Unknown modalities type')

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
            if hyp_params.dataset == 'meld_senti':
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # meld_sentiment
            else:
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # meld_emotion
            text = text * masks_text
            audio = audio * masks_audio
            batch_size = text.size(0)

            net = nn.DataParallel(model) if hyp_params.distribute else model
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    if hyp_params.modalities == 'L':
                        fake_a = trans_net(text, audio, 'train')
                        trans_loss = trans_criterion(fake_a, audio)
                    elif hyp_params.modalities == 'A':
                        fake_l = trans_net(audio, text, 'train')
                        trans_loss = trans_criterion(fake_l, text)
                    else:
                        raise ValueError('Unknown modalities type')
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio)
                else:
                    raise ValueError('Unknown modalities type')
            elif hyp_params.modalities == 'LA':
                preds, _ = net(text, audio)
            else:
                raise ValueError('Unknown modalities type')
            raw_loss = criterion(preds.transpose(1, 2), labels)
            if hyp_params.modalities != 'LA':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss
            combined_loss.backward()

            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()
                else:
                    raise ValueError('Unknown modalities type')

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        model.eval()
        if hyp_params.modalities != 'LA':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                translator.eval()
            else:
                raise ValueError('Unknown modalities type')
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
                if hyp_params.dataset == 'meld_senti':
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # meld_sentiment
                else:
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # meld_emotion
                text = text * masks_text
                audio = audio * masks_audio
                batch_size = text.size(0)

                net = nn.DataParallel(model) if hyp_params.distribute else model
                if hyp_params.modalities != 'LA':
                    if not test:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'L':
                                fake_a = trans_net(text, audio, 'valid')
                                trans_loss = trans_criterion(fake_a, audio)
                            elif hyp_params.modalities == 'A':
                                fake_l = trans_net(audio, text, 'valid')
                                trans_loss = trans_criterion(fake_l, text)
                            else:
                                raise ValueError('Unknown modalities type')
                    else:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'L':
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net(text, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net(text, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            elif hyp_params.modalities == 'A':
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net(audio, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net(audio, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                            else:
                                raise ValueError('Unknown modalities type')
                        else:
                            raise ValueError('Unknown modalities type')
                if hyp_params.modalities != 'LA':
                    if hyp_params.modalities == 'L':
                        preds, _ = net(text, fake_a)
                    elif hyp_params.modalities == 'A':
                        preds, _ = net(fake_l, audio)
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio)
                else:
                    raise ValueError('Unknown modalities type')
                raw_loss = criterion(preds.transpose(1, 2), labels)
                if hyp_params.modalities != 'LA' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss
                total_loss += combined_loss.item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
            mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        else:
            raise ValueError('Unknown modalities type')
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')
    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
    # for epoch in range(1, hyp_params.num_epochs+1):
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()
        train(model, translator, optimizer, criterion)
        val_loss, _, _, _ = evaluate(model, translator, criterion, test=False)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        # print("-"*50)
        # print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}'.format(epoch, duration, val_loss))
        # print("-"*50)

        if val_loss < best_valid:
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if hyp_params.modalities == 'L' or hyp_params.modalities == 'A':
        translator = load_model(hyp_params, name='TRANSLATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, mask = evaluate(model, translator, criterion, test=True)

    acc = eval_meld(results, truths, mask)

    return acc
