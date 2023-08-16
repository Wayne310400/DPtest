import torch
import copy
import random
from itertools import groupby

def aggregate(w, device):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# to get the index of parameter multi-dim parameter to 1 dim 
def FlatSplitParams(model, split_num):
    state_dict = model.state_dict()
    l = [torch.flatten(value) for _, value in state_dict.items()]
    flat_indice = []
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
    return split_index, flat_indice

# Proposed scheme to add dp noise or other clients' weight on client's weight
def ProtectWeight(local_w, noise_slices, weight_slices, split_index, id, flat_indice, device, clip): # add dp noise & other users' weight
    split_num = len(split_index)-1
    # flat local_w which wants to add protect mechanism
    flat_w = torch.cat([torch.flatten(value) for _, value in local_w.items()]).view(-1, 1)
    # bulit add weight sequence
    add_sequence = [(id + i) % split_num for i in range(split_num)]
    # add dp_noise & weight_slice on slice local_w by index
    for i, seq in enumerate(add_sequence):
        if weight_slices[seq] == 'D':
            flat_w[split_index[i]:split_index[i+1]] = flat_w[split_index[i]:split_index[i+1]] / torch.max(torch.FloatTensor([1]).to(device), torch.abs(flat_w[split_index[i]:split_index[i+1]]) / clip)
            flat_w[split_index[i]:split_index[i+1]] += noise_slices[id][i].to(device)
        else:
            flat_w[split_index[i]:split_index[i+1]] = (flat_w[split_index[i]:split_index[i+1]] + weight_slices[seq][i]) / 2
    # unflat protected local_w
    l = [flat_w[s:e] for (s, e) in flat_indice]
    for index, (key, value) in enumerate(local_w.items()):
        if value.shape == torch.Size([]):
            continue
        local_w[key] = l[index].view(*value.shape)

def ProtectWeight2(local_w, noise_slices, weight_slices, split_index, id, flat_indice, device, clip, cut_num, send_rec_parts): # add dp noise & other users' weight
    # flat local_w which wants to add protect mechanism
    flat_w = torch.cat([torch.flatten(value) for _, value in local_w.items()]).view(-1, 1)
    # bulit add weight sequence
    
    part_cat = []
    u_recs = list(filter(lambda send_rec_part: send_rec_part['receiver'] == id, send_rec_parts)) # user 整理他收到哪些部分的模型參數
    # 用戶部分模型參數根據 part_id 進行分組
    for _, category in groupby(sorted(u_recs, key = lambda u_rec:u_rec['part_id']), key=lambda u_rec:u_rec['part_id']):
        list_category = list(category)
        part_cat.append(list_category)
  
    cur_part_id = 0
    for i in range(cut_num):
        if cur_part_id < len(part_cat):
            if i == part_cat[cur_part_id][0]['part_id']: # 收到的話，加起來，再根據收到的份數進行平均
                for part in part_cat[cur_part_id]:
                    flat_w[split_index[i]:split_index[i+1]] = flat_w[split_index[i]:split_index[i+1]] + weight_slices[part['sender']][i]
                flat_w[split_index[i]:split_index[i+1]] = flat_w[split_index[i]:split_index[i+1]] / (len(part_cat[cur_part_id]) + 1)
                cur_part_id += 1
            else: # 如果沒收到那部分的模型參數，用DP噪音保護 
                flat_w[split_index[i]:split_index[i+1]] = flat_w[split_index[i]:split_index[i+1]] / torch.max(torch.FloatTensor([1]).to(device), torch.abs(flat_w[split_index[i]:split_index[i+1]]) / clip)
                flat_w[split_index[i]:split_index[i+1]] += noise_slices[id][i].to(device)
        else: # 如果沒收到那部分的模型參數，用DP噪音保護 
            flat_w[split_index[i]:split_index[i+1]] = flat_w[split_index[i]:split_index[i+1]] / torch.max(torch.FloatTensor([1]).to(device), torch.abs(flat_w[split_index[i]:split_index[i+1]]) / clip)
            flat_w[split_index[i]:split_index[i+1]] += noise_slices[id][i].to(device)

    # unflat protected local_w
    l = [flat_w[s:e] for (s, e) in flat_indice]
    for index, (key, value) in enumerate(local_w.items()):
        if value.shape == torch.Size([]):
            continue
        local_w[key] = l[index].view(*value.shape)