import torch
import numpy as np

from torch import nn
from privacy_meter.audit import Audit
from privacy_meter.dataset import Dataset
from privacy_meter.constants import InferenceGame
from privacy_meter.metric import PopulationMetric
from privacy_meter.model import PytorchModelTensor
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter.information_source import InformationSource
from privacy_meter.hypothesis_test import linear_itp_threshold_func


# all
def sec_func(model, criterion, device, train_data, train_targets, test_data, test_targets, audit_data, audit_targets):
    target_model = PytorchModelTensor(model_obj=model, loss_fn=criterion, device=device, batch_size=10)

    target_dataset = Dataset(
    data_dict={
        "train": {"x": train_data, "y": train_targets},
        "test": {"x": test_data, "y": test_targets},
    },
    default_input="x",
    default_output="y",
    )

    audit_dataset = Dataset(
        data_dict={"train": {"x": audit_data, "y": audit_targets}},
        default_input="x",
        default_output="y",
    )

    target_info_source = InformationSource(
        models=[target_model], 
        datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=[target_model],
        datasets=[audit_dataset]
    )

    metric = PopulationMetric(
                target_info_source=target_info_source,
                reference_info_source=reference_info_source,
                signals=[ModelLoss()],
                hypothesis_test_func=linear_itp_threshold_func,
            )
    audit_obj = Audit(
        metrics=metric,
        inference_game_type=InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        save_logs= False
    )
    audit_obj.prepare()
    return audit_obj.run()[0]

def test(model, device, test_loader):
    model.to(device)    
    # Validate the performance of the model 
    model.eval()
    # Assigning variables for computing loss and accuracy
    loss, acc, criterion = 0, 0, nn.CrossEntropyLoss()

    # Disable gradient calculation to save memory
    with torch.no_grad():
        for data, target in test_loader:
            # Moving data and target to the device
            data, target = data.to(device), target.to(device)

            # Computing output and loss
            output = model(data)
            loss += criterion(output, target).item()

            # Computing accuracy
            pred = output.data.max(1, keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(test_loader)

        # Calculating accuracy
        acc = float(acc) / len(test_loader.dataset)

    # Move the model back to the CPU to save memory
    model.to("cpu")
    return loss, 100. * acc

# dp_train
def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

# proposed_train
def SliceLocalWeight(model, split_index):
    split_num = len(split_index)-1
    state_dict = model.state_dict()
    flat_w = torch.cat([torch.flatten(value) for _, value in state_dict.items()]).view(-1, 1)
    return torch.chunk(flat_w, split_num)

def SliceLocalNoise(sensitivity, noise_scale, clip_num, flat_indice):
    noise_store = torch.tensor([])
    for (s, e) in flat_indice:
        noise_unit = torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * noise_scale, size=(e-s, 1)))
        noise_store = torch.cat((noise_store, noise_unit))
    # cut noise
    weight_slice = torch.chunk(noise_store, clip_num)
    return weight_slice