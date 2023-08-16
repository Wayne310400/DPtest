import torch
import time
import numpy as np
import copy

from torch import nn
from torch.utils.data import TensorDataset
from util import test, sec_func, gaussian_noise, SliceLocalNoise, SliceLocalWeight

def dp_train(model, device, idx, lr, epochs, batch_size, train_loader, test_loader, train_data, train_targets, test_data, test_targets, audit_data, audit_targets, sec_record, q, BATCH_SIZE, clip, sigma, data_size):
    """local model update"""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    start_time = time.time()
    # optimizer = torch.optim.Adam(self.model.parameters())
    
    for epoch_idx in range(epochs):
        # randomly select q fraction samples from data
        # according to the privacy analysis of moments accountant
        # training "Lots" are sampled by poisson sampling
        train_loss = 0
        data_pos = np.where(np.random.rand(len(train_data)) < q)[0]

        sampled_dataset = TensorDataset(train_data[data_pos], train_targets[data_pos])
        sample_data_loader = torch.utils.data.DataLoader(
              dataset=sampled_dataset,
              batch_size=batch_size,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              persistent_workers=True,
              prefetch_factor=16)
        
        optimizer.zero_grad()

        clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for batch_x, batch_y in sample_data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_y = model(batch_x.float())
            loss = criterion(pred_y, batch_y.long())
            
            # bound l2 sensitivity (gradient clipping)
            # clip each of the gradient in the "Lot"
            for i in range(loss.size()[0]):
                loss[i].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                for name, param in model.named_parameters():
                    clipped_grads[name] += param.grad 
                model.zero_grad()
            # Add the loss to the total loss

            train_loss += loss.sum() / len(loss)
                
        # add Gaussian noise
        for name, param in model.named_parameters():
            clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, clip, sigma, device=device)
            
        # scale back
        for name, param in model.named_parameters():
            clipped_grads[name] /= (data_size * q)
        
        for name, param in model.named_parameters():
            param.grad = clipped_grads[name]
        
        # update local model
        optimizer.step()

        loss_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
        # Print the epoch and loss summary
        print(f"ID: {idx} | Epoch: {epoch_idx+1}/{epochs} |", end=" ")
        print(f"Loss: {train_loss/len(train_loader):.4f} |", end=" ")
        print(f"Attack_acc: {100. * loss_audit_results[0].roc_auc:.2f}%")
    if idx == 0:
        loss_audit_results = sec_func(model, criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
        sec_record.append(loss_audit_results[0].roc_auc)
    test_loss, test_acc = test(copy.deepcopy(model), device, test_loader)
    print(f"Test loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("training the target model uses: ", time.time() - start_time)
    return model.state_dict()

def nor_train(model, device, idx, lr, epochs, train_loader, rate_decay):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    if rate_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # if rate_decay:
        #     scheduler.step(train_loss)

    if rate_decay:
        scheduler.step()
        lr[idx] = optimizer.param_groups[0]["lr"]

    return model.state_dict()

def dp_trainv2(model, device, idx, lr, epochs, train_loader, epsilon, delta, glob_epochs, clip, ontrain_frac, rate_decay, dp_strong):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    if rate_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()

            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        if rate_decay:
            scheduler.step(train_loss)
            
    # add Gaussian noise
    model_w = model.state_dict()
    sensitivity = lr[idx] * clip / dp_strong
    times = glob_epochs
    sigma = np.sqrt(2 * np.log(1.25 / (delta / times))) / (epsilon / times) 
    for name, param in model_w.items():
        # print('before: ', param[0][0])
        model_w[name] = param / torch.max(torch.FloatTensor([1]).to(device), torch.abs(param) / clip)
        # print('after: ', model_w[name][0][0])
        model_w[name] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * sigma, size=param.shape)).to(device)
        # print('noise: ', model_w[name][0][0])

    if rate_decay:
        lr[idx] = optimizer.param_groups[0]["lr"]

    return model_w

def proposed_train(model, device, idx, lr, local_e, glob_e, train_loader, epsilon, delta, split_index, flat_indice, num_clients, clip, ontrain_frac, rate_decay, dp_strong):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    if rate_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loop over each epoch
    for epoch_idx in range(local_e):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # if rate_decay:
        #     scheduler.step(train_loss)

    times = glob_e
    sensitivity = lr[idx] * clip  / dp_strong
    sigma = np.sqrt(2 * np.log(1.25 / (delta / times))) / (epsilon / times)

    weight_slice = SliceLocalWeight(copy.deepcopy(model), split_index)
    noise_slice = SliceLocalNoise(sensitivity, sigma, num_clients, flat_indice)

    if rate_decay:
        scheduler.step()
        lr[idx] = optimizer.param_groups[0]["lr"]

    return model.state_dict(), weight_slice, noise_slice

def indust_train(model, device, idx, lr, epochs, train_loader, last_w, flat_indice, clip, delta, glob_epochs, epsilon, ontrain_frac, rate_decay, dp_strong):
    global_weight = copy.deepcopy(model.state_dict())
    # random select partial global parameter to update local model
    if last_w[idx] != []:
        glo_w = copy.deepcopy(global_weight)
        flat_last_w = torch.cat([torch.flatten(value) for _, value in last_w[idx].items()]).view(-1, 1).to(device)
        flat_glob_w = torch.cat([torch.flatten(value) for _, value in glo_w.items()]).view(-1, 1).to(device)
        rand_index = np.random.choice(range(len(flat_glob_w)), int(len(flat_glob_w) * 0.7), replace=False)
        flat_last_w[rand_index] = flat_glob_w[rand_index]
        # unflat protected local_w
        l = [flat_last_w[s:e] for (s, e) in flat_indice]
        for index, (key, value) in enumerate(glo_w.items()):
            if value.shape == torch.Size([]):
                continue
            glo_w[key] = l[index].view(*value.shape)
        model.load_state_dict(glo_w)

    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    if rate_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        if rate_decay:
            scheduler.step(train_loss)

    # select partial parameters to 0, the others remain origin value
    local_weight = copy.deepcopy(model.state_dict())
    times = glob_epochs
    sensitivity = lr[idx] * clip  / dp_strong
    sigma = np.sqrt(2 * np.log(1.25 / (delta / times))) / (epsilon / times)
    for name, param in local_weight.items():
        local_weight[name] = param / torch.max(torch.FloatTensor([1]).to(device), torch.abs(param) / clip)
        local_weight[name] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * sigma, size=param.shape)).to(device)
    flat_l = torch.cat([torch.flatten(value) for _, value in local_weight.items()]).view(-1, 1).to(device)
    flat_g = torch.cat([torch.flatten(value) for _, value in global_weight.items()]).view(-1, 1).to(device)
    value, index = torch.topk(torch.abs(torch.sub(flat_l, flat_g)), int(len(flat_l) * 0.7), dim=0, largest=False) # choose smallest
    flat_l[index] = flat_g[index]
    l = [flat_l[s:e] for (s, e) in flat_indice]
    for index, (key, value) in enumerate(local_weight.items()):
        if(value.shape == torch.Size([])):
            continue
        local_weight[key] = l[index].view(*value.shape)
            
    if rate_decay:
        lr[idx] = optimizer.param_groups[0]["lr"]

    return local_weight, model.state_dict()

def proposed_train2(model, device, idx, lr, local_e, glob_e, train_loader, epsilon, delta, split_index, flat_indice, clip, rate_decay, dp_strong, cut_num):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    if rate_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loop over each epoch
    for epoch_idx in range(local_e):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # if rate_decay:
        #     scheduler.step(train_loss)

    times = glob_e
    sensitivity = lr[idx] * clip  / dp_strong
    sigma = np.sqrt(2 * np.log(1.25 / (delta / times))) / (epsilon / times)

    weight_slice = SliceLocalWeight(copy.deepcopy(model), split_index)
    noise_slice = SliceLocalNoise(sensitivity, sigma, cut_num, flat_indice)

    if rate_decay:
        scheduler.step()
        lr[idx] = optimizer.param_groups[0]["lr"]

    return model.state_dict(), weight_slice, noise_slice