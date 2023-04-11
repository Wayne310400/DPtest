#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k] * size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

def FlatSplitParams(global_w, split_num):
    state_dict = copy.deepcopy(global_w)
    l = [torch.flatten(value) for key, value in state_dict.items()]
    flat_indice, drop_protect = [], []
    s = 0
    for p in l:
        size = p.shape[0]
        flat_indice.append((s, s+size))
        s += size
    flat_w = torch.cat(l).view(-1, 1)
    split_w = torch.chunk(flat_w, split_num)
    split_index = [0]
    for i in split_w:
        split_index.append(split_index[-1] + len(i))

    for k, v in state_dict.items():
        state_dict[k] = torch.zeros(v.shape)
    for i in range(split_num):
        drop_protect.append(state_dict)
    return drop_protect, split_index, flat_indice

def SliceLocalWeight(local_w, split_index):
    split_num = len(split_index)-1
    state_dict = copy.deepcopy(local_w)
    flat_w = torch.cat([torch.flatten(value) for key, value in state_dict.items()]).view(-1, 1)
    return torch.chunk(flat_w, split_num)

def ProtectWeight(local_w, dp_noise, weight_slices, split_index, id, flat_indice): # add dp noise & other users' weight
    split_num = len(split_index)-1
    # flat local_w which wants to add protect mechanism
    flat_w = torch.cat([torch.flatten(value) for key, value in local_w.items()]).view(-1, 1)
    # bulit add weight sequence
    add_sequence = [(id + i) % split_num for i in range(split_num)]
    # add dp_noise & weight_slice on slice local_w by index
    for i, seq in enumerate(add_sequence):
        if weight_slices[seq] == 'D':
            flat_w[split_index[i]:split_index[i+1]] += dp_noise[id][i]
        else:
            flat_w[split_index[i]:split_index[i+1]] = (flat_w[split_index[i]:split_index[i+1]] + weight_slices[seq][i]) / 2
    # unflat protected local_w
    l = [flat_w[s:e] for (s, e) in flat_indice]
    for index, (key, value) in enumerate(local_w.items()):
        local_w[key] = l[index].view(*value.shape)
    return local_w