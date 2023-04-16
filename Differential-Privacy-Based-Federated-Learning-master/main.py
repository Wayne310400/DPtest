#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from utils.analysis import security_analysis
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg, FlatSplitParams, SliceLocalWeight, ProtectWeight
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':
    # parse args

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    all_clients = list(range(args.num_users))

    # get indice & dropout users' protect weight noise(is zero)
    drop_protect, split_index, flat_indice = FlatSplitParams(w_glob, args.num_users)

    sec_train = copy.deepcopy(dataset_train)
    sec_test = copy.deepcopy(dataset_test)

    # training
    acc_test, loss_test, time_test, security_test = [], [], [], []
    if args.serial:
        clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]

    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols, weight_slices, noise_slices, nondrop_index, drop_index = [], [], [], [], [], [], []
        m = max(int(args.frac * args.num_users), 1)

        # choice dropout users
        drop_users = random.sample(range(args.num_users), int(args.num_users * args.drop))
        nondrop_users = [x for x in range(args.num_users) if x not in drop_users]
        for i in nondrop_users:
            nondrop_index.append([split_index[i], split_index[i+1]])

        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        begin_index = iter % (1 / args.frac)
        idxs_users = all_clients[int(begin_index * args.num_users * args.frac):
                                   int((begin_index + 1) * args.num_users * args.frac)]
        for idx in idxs_users:
            if idx in drop_users:
                weight_slices.append("D") # mark dropout users' weight
                noise_slices.append("D") # mark dropout users' noise
                continue
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.dp_mechanism != 'no_dp': # add DP noise or get DP noise slice 
                noisy_w, noise_slice = local.add_noise(copy.deepcopy(w), flat_indice)
                noise_slices.append(noise_slice)
            else:
                noisy_w = w
            if args.dp_mechanism == 'Partial':
                weight_slice = SliceLocalWeight(w, split_index) # divide local weight to several unit
                weight_slices.append(weight_slice)
            w_locals.append(copy.deepcopy(noisy_w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # if get other users' partial weight add them and divide 2 by index; if not, add DP noise bu index
        if args.dp_mechanism == 'Partial':
            for i, id in enumerate(nondrop_users):
                ProtectWeight(args, w_locals[i], noise_slices, weight_slices, split_index, id, flat_indice) # this function will directly change w_local

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        time_t = time.time() - t_start
        # print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s, dropout users: {}".format(iter, acc_t, time_t, drop_users))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
        time_test.append(time_t)

        # choose user 0 to analysis security
        chosen_one = nondrop_users[0]
        train_index = dict_users[chosen_one]
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_analysis = CNNCifar(args=args)
        elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
            net_analysis = CNNMnist(args=args)
        else:
            print("Model selection error!")
        net_analysis.load_state_dict({k.replace('_module.', ''):v for k, v in copy.deepcopy(w_locals[0]).items()})
        audit_result = security_analysis(args, sec_train, sec_test, train_index, net_analysis)
        security_test.append(audit_result.roc_auc)

        print("Round {:3d},Testing accuracy: {:.2f}, Time:  {:.2f}s, Attack accuracy: {:.2f}".format(iter, acc_t, time_t, audit_result.roc_auc))

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    if not os.path.exists(rootpath + '/acc'):
        os.makedirs(rootpath + '/acc')
    if not os.path.exists(rootpath + '/loss'):
        os.makedirs(rootpath + '/loss')
    if not os.path.exists(rootpath + '/time'):
        os.makedirs(rootpath + '/time')
    if not os.path.exists(rootpath + '/security'):
        os.makedirs(rootpath + '/security')
    accfile = open(rootpath + '/acc' + '/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                    format(args.dataset, args.model, args.epochs, args.iid,
                    args.dp_mechanism, args.dp_epsilon, args.drop), "w")
    lossfile = open(rootpath + '/loss' + '/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}.dat'.
                    format(args.dataset, args.model, args.epochs, args.iid,
                    args.dp_mechanism, args.dp_epsilon, args.drop), "w")
    timefile = open(rootpath + '/time' + '/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}_time_{:.2f}.dat'.
                    format(args.dataset, args.model, args.epochs, args.iid,
                    args.dp_mechanism, args.dp_epsilon, args.drop, sum(time_test)), "w")
    securityfile = open(rootpath + '/security' + '/fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_drop_{}_time_{:.2f}.dat'.
                        format(args.dataset, args.model, args.epochs, args.iid,
                        args.dp_mechanism, args.dp_epsilon, args.drop), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    for lo in loss_test:
        slo = str(lo)
        lossfile.write(slo)
        lossfile.write('\n')
    lossfile.close()

    for ti in time_test:
        sti = str(ti)
        timefile.write(sti)
        timefile.write('\n')
    timefile.close()

    for se in security_test:
        sse = str(se)
        securityfile.write(sse)
        securityfile.write('\n')
    securityfile.close()

    # plot loss & accuracy curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/acc' + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_drop_{}_acc.png'.format(
            args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.drop))
    
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath + '/loss' + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_drop_{}_loss.png'.format(
            args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.drop))
    
    plt.figure()
    plt.plot(range(len(security_test)), security_test)
    plt.ylabel('attack accuracy')
    plt.savefig(rootpath + '/security' + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_drop_{}_security.png'.format(
            args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.drop))