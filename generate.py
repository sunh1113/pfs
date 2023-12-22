import torch
import numpy as np
from os.path import join
from utils.name import get_name
from utils.settings import DATASETTINGS
from datasets import transform_set, build_data
from models import build_model
from attacks import build_trigger

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime
import sys


def generate(opts):
    print(''.join(['-'] * 100))
    name = get_name(opts)
    print('name: ', name)
    print(''.join(['-'] * 100))

    dast = DATASETTINGS[opts.data_name]
    train_transform = transform_set(opts.transform_id, True, dast['img_size'], dast['crop_pad'], dast['flip'])
    val_transform = transform_set(opts.transform_id, False, dast['img_size'], dast['crop_pad'], dast['flip'])

    if opts.attack_name == 'opti':
        loads = np.load('./triggers/c10_pgd_random_0.npy', allow_pickle=True).item()
        n = loads['n']
        trigger = build_trigger(opts.attack_name, dast['img_size'], dast['num_data'], n, 0, opts.attack_target)
    else:
        trigger = build_trigger(opts.attack_name, dast['img_size'], dast['num_data'], None, 0, opts.attack_target)

    train_data = build_data(opts.data_name, opts.data_path, True, trigger, train_transform)
    val_data = build_data(opts.data_name, opts.data_path, False, trigger, val_transform)
    idx_target = np.arange(len(train_data))[np.array(train_data.targets) == opts.attack_target]

    poison_num = int(len(train_data) * opts.poison_ratio)
    print('{} poisoned samples'.format(poison_num))

    # opts.fus_n_iter = opts.fus_n_iter
    shuffle = []
    idx_top = []
    sample_idx = []

    if opts.select_name == 'rand' or opts.select_name == 'fus':
        if opts.select_name == 'rand':
            opts.fus_n_iter = 1
        shuffle = np.arange(len(train_data))[np.array(train_data.targets) != opts.attack_target]
        np.random.shuffle(shuffle)
        sample_idx = shuffle[:poison_num]

    if opts.select_name == 'pfs' or opts.select_name == 'pf':
        if opts.select_name == 'pfs':
            opts.fus_n_iter = 1
        sim_score = np.load('./distance/r18_c10_blend.npy'.format(opts.data_name, opts.attack_name))
        sim_idx = np.argsort(sim_score)
        idx_untarget = np.setdiff1d(sim_idx, idx_target, assume_unique=True)
        print(idx_untarget, len(idx_untarget))


        top = np.arange(opts.m * poison_num)
        np.random.shuffle(top)
        idx_top = idx_untarget[top]
        print('len idx_top:', len(idx_top))
        sample_idx = idx_top[:poison_num]
        print('len sample_idx:', len(sample_idx))


    # if opts.select_name == 'transfer':
    #     opts.fus_n_iter = 1
    #     sample_idx = np.load('./npy/c10_blend_fus_r18_0.01_0.7_15_0_{}_0.npy')
    #     print('transfer, {} poisoned samples'.format(len(sample_idx)))

    results = {}
    sample_idx_best, best_back_acc = [], 0


    starttime_1 = datetime.datetime.now()
    for n in range(opts.fus_n_iter):
        print('--------------Train with {:2d} iteration'.format(n) + ''.join(['-'] * 15))
        results[n] = {'loss': [], 'train_acc': [], 'val_acc': [], 'back_acc': []}

        train_data = build_data(opts.data_name, opts.data_path, True, trigger, train_transform)
        train_data.data = np.concatenate((train_data.data, train_data.data[sample_idx]), axis=0)
        train_data.targets = train_data.targets + [train_data.targets[i] for i in sample_idx]
        train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False, num_workers=2)

        model = build_model(opts.model_name, dast['num_classes'])
        model = model.to(opts.device) 
        ###############################################################################################################
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        ###############################################################################################################
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, dast['decay_steps'], 0.1)
        criterion = nn.CrossEntropyLoss().to(opts.device)

        measure = []
        for epoch in range(dast['epochs']):
            trigger.set_mode(0), model.train()
            correct, total, ps, ds, vs = 0, 0, [], [], []
            for x, y, b, s, d in train_loader:
                x, y, b, s, d = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device), d.to(
                    opts.device)
                optimizer.zero_grad()
                f, p = model(x)
                loss = criterion(p, y)
                loss.backward()
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
                optimizer.step()
                results[n]['loss'].append(loss.item())
                ps.append((p == y)[d >= dast['num_data']].long().detach().cpu().numpy())
                ds.append(d[d >= dast['num_data']].detach().cpu().numpy())
            scheduler.step()
            train_acc = correct / (total + 1e-12)
            results[n]['train_acc'].append(train_acc)

            ps, ds = np.concatenate(ps, axis=0), np.concatenate(ds, axis=0)
            ps = ps[np.argsort(ds)]
            measure.append(ps[:, np.newaxis])

            trigger.set_mode(1), model.eval()
            correct, total = 0, 0
            for x, y, _, _, _ in val_loader:
                x, y = x.to(opts.device), y.to(opts.device)
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            val_acc = correct / (total + 1e-12)
            results[n]['val_acc'].append(val_acc)

            trigger.set_mode(2), model.eval()
            correct, total = 0, 0
            for x, y, _, s, _ in val_loader:
                x, y, s = x.to(opts.device), y.to(opts.device), s.to(opts.device)
                idx = s != opts.attack_target
                x, y, s = x[idx, :, :, :], y[idx], s[idx]
                if x.shape[0] == 0: continue
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            back_acc = correct / (total + 1e-12)
            results[n]['back_acc'].append(back_acc)

            print('epoch: {:2d}, train_acc: {:.3f}, val_acc: {:.3f}, back_acc: {:.3f}'.format(epoch, train_acc, val_acc,
                                                                                              back_acc))

        if results[n]['back_acc'][-1] > best_back_acc:  # save new best
            print('update new sample idx with back acc: {:.3f}'.format(results[n]['back_acc'][-1]))
            best_back_acc = results[n]['back_acc'][-1]
            sample_idx_best = np.copy(sample_idx)

        measure = np.concatenate(measure, axis=1)
        diff_measure = measure[:, 1:] - measure[:, :-1]
        score = np.sum(diff_measure == -1, axis=1)
        score_idx = np.argsort(score)[::-1]  # sort from large to small

        results[n]['measure'] = measure
        results[n]['score'] = score
        results[n]['sample_idx'] = np.copy(sample_idx)

        sample_idx = sample_idx[score_idx]
        sample_idx = sample_idx[:int(len(sample_idx) * opts.fus_alpha)]

        if opts.select_name == 'fus':
            np.random.shuffle(shuffle)
            sample_idx = np.concatenate((sample_idx, shuffle[:(poison_num - len(sample_idx))]), axis=0)

        if opts.select_name == 'pf':
            num_next = poison_num - len(sample_idx)
            next_start_idx = poison_num * (1 + n * (1 - opts.fus_alpha))
            next_start_idx = int(round(next_start_idx))
            print('num_next, next_start_idx: ', num_next, next_start_idx)
            shu = np.arange(next_start_idx, next_start_idx + num_next)
            next_sample_idx = idx_top[shu]
            sample_idx = np.concatenate((sample_idx, next_sample_idx), axis=0)
            print('len sample_idx new: ', len(sample_idx))

    print('Best backdoor acc: {:.3f}'.format(best_back_acc))
    endtime_1 = datetime.datetime.now()
    print('run time: ', endtime_1 - starttime_1)


    print('save poisoned sample idx and running results')
    np.save(join(opts.sample_idx_path, '{}.npy'.format(name)), sample_idx_best)  # save the selected poisoned sample idx
    np.save(join(opts.result_path, '{}.npy'.format(name)), results)


