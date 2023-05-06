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

def nor_train(model, device, idx, lr, epochs, train_loader):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9995)
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)
            # Cast target to long tensor
            target = target

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
            # scheduler.step()

            # Add the loss to the total loss
            train_loss += loss.item()
        # scheduler.step()
            
        # loss_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
        # # Print the epoch and loss summary
        # print(f"ID: {idx} | Epoch: {epoch_idx+1}/{epochs} |", end=" ")
        # print(f"Loss: {train_loss/len(train_loader):.4f} |", end=" ")
        # print(f"Attack_acc: {100. * loss_audit_results[0].roc_auc:.2f}%")
    # if idx == 0:
    #     loss_audit_results = sec_func(model, criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
    #     sec_record.append(loss_audit_results[0].roc_auc)
    # test_loss, test_acc = test(copy.deepcopy(model), device, test_loader)
    # print(f"Test loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    # scheduler.step()
    # lr[idx] = scheduler.get_last_lr()[0]
    return model.state_dict()

def dp_trainv2(model, device, idx, lr, epochs, train_loader, train_data, epsilon, delta, glob_epochs, clip):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            for param in model.parameters():
                param.accumulated_grads = []
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)
            # Cast target to long tensor
            target = target

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
            scheduler.step()

            # Add the loss to the total loss
            train_loss += loss.item()
            
    #     loss_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
    #     # Print the epoch and loss summary
    #     print(f"ID: {idx} | Epoch: {epoch_idx+1}/{epochs} |", end=" ")
    #     print(f"Loss: {train_loss/len(train_loader):.4f} |", end=" ")
    #     print(f"Attack_acc: {100. * loss_audit_results[0].roc_auc:.2f}%")
    # add Gaussian noise
    model_w = model.state_dict()
    # sensitivity = 5 * lr * clip  / len(train_data)
    sensitivity = 2 * lr[idx] * clip  / len(train_data)
    sigma = np.sqrt(2 * np.log(1.25 / (delta / glob_epochs))) / (epsilon / glob_epochs) 
    # sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    for name, param in model_w.items():
        model_w[name] = param / torch.max(torch.FloatTensor([1]).to(device), torch.abs(param) / clip)
        # model_w[name] += torch.normal(0, sensitivity * sigma, param.shape).to(device)
        model_w[name] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * sigma, size=param.shape)).to(device)
    # if idx == 0:
    #     loss_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
    #     sec_record.append(loss_audit_results[0].roc_auc)
    # loss_noisy_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
    # test_loss, test_acc = test(copy.deepcopy(model), device, test_loader)
    # print(f"Test loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Attack Acc: {100 * loss_noisy_audit_results[0].roc_auc:.2f}%")
    # scheduler.step()
    lr[idx] = scheduler.get_last_lr()[0]
    return model_w

def proposed_train(model, device, idx, lr, local_e, glob_e, train_loader, train_data, epsilon, delta, split_index, flat_indice, num_clients, clip):
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr[idx], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    # Loop over each epoch
    for epoch_idx in range(local_e):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)
            # Cast target to long tensor
            target = target

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
            
    #     loss_audit_results = sec_func(copy.deepcopy(model), criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets)
    #     # Print the epoch and loss summary
    #     print(f"ID: {idx} | Epoch: {epoch_idx+1}/{local_e} |", end=" ")
    #     print(f"Loss: {train_loss/len(train_loader):.4f} |", end=" ")
    #     print(f"Attack_acc: {100. * loss_audit_results[0].roc_auc:.2f}%")

    # test_loss, test_acc = test(copy.deepcopy(model), device, test_loader)
    # print(f"Test loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    sensitivity = lr[idx] * clip  / len(train_data)
    sigma = np.sqrt(2 * np.log(1.25 / (delta / glob_e))) / (epsilon / glob_e)

    weight_slice = SliceLocalWeight(copy.deepcopy(model), split_index)
    noise_slice = SliceLocalNoise(sensitivity, sigma, num_clients, flat_indice)

    # scheduler.step()
    # lr[idx] = scheduler.get_last_lr()[0]

    return model.state_dict(), weight_slice, noise_slice