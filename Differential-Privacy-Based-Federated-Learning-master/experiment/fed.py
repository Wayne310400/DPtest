import torch
import copy

def aggregate(w_locals, clients, device):
    """FedAvg"""
    new_w = copy.deepcopy(w_locals[0])
    for name in new_w:
        new_w[name] = torch.zeros(new_w[name].shape).to(device)
    for idx in range(clients):
        for name in new_w:
            new_w[name] += w_locals[idx][name].to(device) * (1 / clients)
    return copy.deepcopy(new_w)

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
def ProtectWeight(local_w, noise_slices, weight_slices, split_index, id, flat_indice, device): # add dp noise & other users' weight
    split_num = len(split_index)-1
    # flat local_w which wants to add protect mechanism
    flat_w = torch.cat([torch.flatten(value) for _, value in local_w.items()]).view(-1, 1)
    # bulit add weight sequence
    add_sequence = [(id + i) % split_num for i in range(split_num)]
    # add dp_noise & weight_slice on slice local_w by index
    for i, seq in enumerate(add_sequence):
        if weight_slices[seq] == 'D':
            flat_w[split_index[i]:split_index[i+1]] += noise_slices[id][i].to(device)
        else:
            flat_w[split_index[i]:split_index[i+1]] = (flat_w[split_index[i]:split_index[i+1]] + weight_slices[seq][i]) / 2
    # unflat protected local_w
    l = [flat_w[s:e] for (s, e) in flat_indice]
    for index, (key, value) in enumerate(local_w.items()):
        local_w[key] = l[index].view(*value.shape)