import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from src.eval_metrics import *

####################################################################
#
# Construct the model
#
####################################################################

text_dim, audio_dim, video_dim = 300, 81, 371


def crop(a): return a[:, :, :, 0:text_dim], a[:, :, :, text_dim:(text_dim + audio_dim)], a[:, :, :,
                                                                                         (text_dim + audio_dim)::]


def initiate(hyp_params, train_loader, valid_loader, test_loader, onlypunch=False):
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'A':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'V':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LA':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'AV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        else:
            raise ValueError('Unknown modalities type')
        trans_criterion = getattr(nn, 'MSELoss')()
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            settings = {'model': model,
                        'translator1': translator1,
                        'translator2': translator2,
                        'translator1_optimizer': translator1_optimizer,
                        'translator2_optimizer': translator2_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            settings = {'model': model,
                        'translator': translator,
                        'translator_optimizer': translator_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        else:
            raise ValueError('Unknown modalities type')
    elif hyp_params.modalities == 'LAV':
        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler}
    else:
        raise ValueError('Unknown modalities type')
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, onlypunch)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, onlypunch=False):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    if hyp_params.modalities != 'LAV':
        trans_criterion = settings['trans_criterion']
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            translator1 = settings['translator1']
            translator2 = settings['translator2']
            translator1_optimizer = settings['translator1_optimizer']
            translator2_optimizer = settings['translator2_optimizer']
            translator = (translator1, translator2)
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']
        else:
            raise ValueError('Unknown modalities type')
    else:
        translator = None

    def train(model, translator, optimizer, criterion, onlypunch=False):
        if isinstance(translator, tuple):
            translator1, translator2 = translator
        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator1.train()
                translator2.train()
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                translator.train()
            else:
                raise ValueError('Unknown modalities type')
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (x_c, x_p, eval_attr) in enumerate(train_loader):
            x_p = torch.unsqueeze(x_p, dim=1)
            combined = x_p if onlypunch else torch.cat([x_c, x_p], dim=1)
            t, a, v = crop(combined)
            bs, segs, lens, _ = a.shape
            audio, text, vision = a.reshape(bs, segs * lens, -1), t.reshape(bs, segs * lens, -1), v.reshape(bs,
                                                                                                            segs * lens,
                                                                                                            -1)

            model.zero_grad()
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    translator1.zero_grad()
                    translator2.zero_grad()
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                    translator.zero_grad()
                else:
                    raise ValueError('Unknown modalities type')

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

            batch_size = text.size(0)

            net = nn.DataParallel(model) if hyp_params.distribute else model
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                    trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                    if hyp_params.modalities == 'L':
                        fake_a = trans_net1(text, audio, 'train')
                        fake_v = trans_net2(text, vision, 'train')
                        trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'A':
                        fake_l = trans_net1(audio, text, 'train')
                        fake_v = trans_net2(audio, vision, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'V':
                        fake_l = trans_net1(vision, text, 'train')
                        fake_a = trans_net2(vision, audio, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    if hyp_params.modalities == 'LA':
                        fake_v = trans_net((text, audio), vision, 'train')
                        trans_loss = trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'LV':
                        fake_a = trans_net((text, vision), audio, 'train')
                        trans_loss = trans_criterion(fake_a, audio)
                    elif hyp_params.modalities == 'AV':
                        fake_l = trans_net((audio, vision), text, 'train')
                        trans_loss = trans_criterion(fake_l, text)
                    else:
                        raise ValueError('Unknown modalities type')
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a, fake_v)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio, fake_v)
                elif hyp_params.modalities == 'V':
                    preds, _ = net(fake_l, fake_a, vision)
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio, fake_v)
                elif hyp_params.modalities == 'LV':
                    preds, _ = net(text, fake_a, vision)
                elif hyp_params.modalities == 'AV':
                    preds, _ = net(fake_l, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')
            elif hyp_params.modalities == 'LAV':
                preds, _ = net(text, audio, vision)
            else:
                raise ValueError('Unknown modalities type')

            raw_loss = criterion(preds, eval_attr)
            if hyp_params.modalities != 'LAV':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss
            combined_loss.backward()

            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    torch.nn.utils.clip_grad_norm_(translator1.parameters(), hyp_params.clip)
                    torch.nn.utils.clip_grad_norm_(translator2.parameters(), hyp_params.clip)
                    translator1_optimizer.step()
                    translator2_optimizer.step()
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
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

    def evaluate(model, translator, criterion, test=False, onlypunch=False):
        if isinstance(translator, tuple):
            translator1, translator2 = translator
        model.eval()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator1.eval()
                translator2.eval()
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                translator.eval()
            else:
                raise ValueError('Unknown modalities type')
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (x_c, x_p, eval_attr) in enumerate(loader):
                x_p = torch.unsqueeze(x_p, dim=1)
                combined = x_p if onlypunch else torch.cat([x_c, x_p], dim=1)
                t, a, v = crop(combined)
                bs, segs, lens, _ = a.shape
                audio, text, vision = a.reshape(bs, segs * lens, -1), t.reshape(bs, segs * lens, -1), v.reshape(bs,
                                                                                                                segs * lens,
                                                                                                                -1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

                batch_size = text.size(0)

                net = nn.DataParallel(model) if hyp_params.distribute else model
                if hyp_params.modalities != 'LAV':
                    if not test:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                            if hyp_params.modalities == 'L':
                                fake_a = trans_net1(text, audio, 'valid')
                                fake_v = trans_net2(text, vision, 'valid')
                                trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'A':
                                fake_l = trans_net1(audio, text, 'valid')
                                fake_v = trans_net2(audio, vision, 'valid')
                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'V':
                                fake_l = trans_net1(vision, text, 'valid')
                                fake_a = trans_net2(vision, audio, 'valid')
                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                            else:
                                raise ValueError('Unknown modalities type')
                        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'LA':
                                fake_v = trans_net((text, audio), vision, 'valid')
                                trans_loss = trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'LV':
                                fake_a = trans_net((text, vision), audio, 'valid')
                                trans_loss = trans_criterion(fake_a, audio)
                            elif hyp_params.modalities == 'AV':
                                fake_l = trans_net((audio, vision), text, 'valid')
                                trans_loss = trans_criterion(fake_l, text)
                            else:
                                raise ValueError('Unknown modalities type')
                    else:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                            if hyp_params.modalities == 'L':
                                fake_a = torch.Tensor().cuda()
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net1(text, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net1(text, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(text, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(text, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'A':
                                fake_l = torch.Tensor().cuda()
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(audio, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(audio, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(audio, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(audio, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'V':
                                fake_l = torch.Tensor().cuda()
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(vision, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(vision, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net2(vision, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net2(vision, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            else:
                                raise ValueError('Unknown modalities type')
                        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'LA':
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net((text, audio), vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net((text, audio), fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'LV':
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net((text, vision), audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net((text, vision), fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            elif hyp_params.modalities == 'AV':
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net((audio, vision), text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net((audio, vision), fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                            else:
                                raise ValueError('Unknown modalities type')
                        else:
                            raise ValueError('Unknown modalities type')
                if hyp_params.modalities != 'LAV':
                    if hyp_params.modalities == 'L':
                        preds, _ = net(text, fake_a, fake_v)
                    elif hyp_params.modalities == 'A':
                        preds, _ = net(fake_l, audio, fake_v)
                    elif hyp_params.modalities == 'V':
                        preds, _ = net(fake_l, fake_a, vision)
                    elif hyp_params.modalities == 'LA':
                        preds, _ = net(text, audio, fake_v)
                    elif hyp_params.modalities == 'LV':
                        preds, _ = net(text, fake_a, vision)
                    elif hyp_params.modalities == 'AV':
                        preds, _ = net(fake_l, audio, vision)
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'LAV':
                    preds, _ = net(text, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')
                raw_loss = criterion(preds, eval_attr)
                if hyp_params.modalities != 'LAV' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss
                total_loss += combined_loss.item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            mgm_parameter1 = sum([param.nelement() for param in translator1.parameters()])
            mgm_parameter2 = sum([param.nelement() for param in translator2.parameters()])
            mgm_parameter = mgm_parameter1 + mgm_parameter2
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
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
        train(model, translator, optimizer, criterion, onlypunch=onlypunch)
        val_loss, _, _ = evaluate(model, translator, criterion, test=False, onlypunch=onlypunch)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        # print("-"*50)
        # print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}'.format(epoch, duration, val_loss))
        # print("-"*50)

        if val_loss < best_valid:
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                save_model(hyp_params, translator1, name='TRANSLATOR_1')
                save_model(hyp_params, translator2, name='TRANSLATOR_2')
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
        translator1 = load_model(hyp_params, name='TRANSLATOR_1')
        translator2 = load_model(hyp_params, name='TRANSLATOR_2')
        translator = (translator1, translator2)
    elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
        translator = load_model(hyp_params, name='TRANSLATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, translator, criterion, test=True, onlypunch=onlypunch)
    acc = eval_ur_funny(results, truths)

    return acc
