import warnings
warnings.simplefilter("ignore", UserWarning)
import torchvision
import torch
import numpy as np
import os

from ast import List
from torch import nn

from privacy_meter.audit import Audit
from privacy_meter.metric import PopulationMetric
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter.hypothesis_test import linear_itp_threshold_func
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModelTensor

def get_dataset_subset(dataset: torchvision.datasets, index: List(int)):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    data = (
        torch.from_numpy(dataset.data[index]).unsqueeze(1).float() / 255
    )  # channel first
    targets = list(np.array(dataset.targets)[index])
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def security_analysis(args, train_x, test_x, train_index, model):
    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    if not os.path.exists(rootpath + '/ml'):
        os.makedirs(rootpath + '/ml')

    all_data = train_x
    all_features = np.concatenate([train_x.data, test_x.data], axis=0)
    all_targets = np.concatenate([train_x.targets, test_x.targets], axis=0)
    all_data.data = all_features
    all_data.targets = all_targets

    num_test_points = len(train_index)
    num_population_points = 2 * num_test_points

    all_index = np.arange(len(all_data))
    test_index = np.random.choice([i for i in all_index if i not in train_index], num_test_points, replace=False)
    population_index = np.random.choice([i for i in all_index if i not in train_index and i not in test_index], num_population_points, replace=False)

    target_model = PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(), device=args.device, batch_size=10)

    train_data, train_targets = get_dataset_subset(all_data, train_index)
    test_data, test_targets = get_dataset_subset(all_data, test_index)
    audit_data, audit_targets = get_dataset_subset(all_data, population_index)
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
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        save_logs=False
    )
    audit_obj.prepare()
    loss_audit_results = audit_obj.run()[0]
 
    return loss_audit_results[0]

# import numpy as np

# import torch.nn.functional as F
# from torch import nn
# import torchvision
# import os
# import pickle
# from ast import List
# import time
# import numpy as np
# import torch
# import torchvision
# from torchvision import transforms

# from privacy_meter.audit import Audit, MetricEnum
# from privacy_meter.metric import PopulationMetric
# from privacy_meter.information_source_signal import ModelGradientNorm, ModelGradient, ModelLoss
# from privacy_meter.hypothesis_test import linear_itp_threshold_func
# from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
# from privacy_meter.constants import InferenceGame
# from privacy_meter.dataset import Dataset
# from privacy_meter.information_source import InformationSource
# from privacy_meter.model import PytorchModelTensor

# def get_dataset_subset(dataset: torchvision.datasets, index: List(int)):
#     """Get a subset of the dataset.

#     Args:
#         dataset (torchvision.datasets): Whole dataset.
#         index (list): List of index.
#     """
#     assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
#     print(torch.from_numpy(dataset.data[index]).shape)
#     print(torch.from_numpy(dataset.data[index]).unsqueeze(1).float().shape)
#     data = (
#         torch.from_numpy(dataset.data[index]).unsqueeze(1).float() / 255
#     )  # channel first
#     targets = list(np.array(dataset.targets)[index])
#     targets = torch.tensor(targets, dtype=torch.long)
#     return data, targets

# def security_analysis(args, train_x, test_x, train_index, model):
#     device = 'cpu'
#     criterion = nn.CrossEntropyLoss()
#     num_train_points = 600
#     num_test_points = 600
#     num_population_points = 1200
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     all_data = torchvision.datasets.MNIST(
#         root='.', train=True, download=True, transform=transform
#     )
#     test_data = torchvision.datasets.MNIST(
#         root='.', train=False, download=True, transform=transform
#     )
#     all_features = np.concatenate([train_x.data, test_x.data], axis=0)
#     all_targets = np.concatenate([train_x.targets, test_x.targets], axis=0)
#     # if torch.equal(train_x[0][0], all_data[0][0]):
#     #     print(train_x[0][0])
#     #     print("Good1")
#     # if torch.equal(train_x[1][0], all_data[1][0]):
#     #     print("Good2")
#     # if torch.equal(train_x[2][0], all_data[2][0]):
#     #     print("Good3")

#     all_data.data = all_features
#     all_data.targets = all_targets

#     all_index = np.arange(len(all_data))
#     test_index = np.random.choice([i for i in all_index if i not in train_index], num_test_points, replace=False)
#     population_index = np.random.choice([i for i in all_index if i not in train_index and i not in test_index], num_population_points, replace=False)

#     class Net(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.conv2_drop = nn.Dropout2d()
#             self.fc1 = nn.Linear(320, 50)
#             self.fc2 = nn.Linear(50, 10)

#         def forward(self, x):
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             x = self.fc2(x)
#             return F.log_softmax(x, dim=1)
#     target_model = PytorchModelTensor(model_obj=Net(), loss_fn=criterion, device=device,batch_size=10)

#     train_data, train_targets = get_dataset_subset(all_data, train_index)
#     test_data, test_targets = get_dataset_subset(all_data, test_index)
#     audit_data, audit_targets = get_dataset_subset(all_data, population_index)
#     target_dataset = Dataset(
#         data_dict={
#             "train": {"x": train_data, "y": train_targets},
#             "test": {"x": test_data, "y": test_targets},
#         },
#         default_input="x",
#         default_output="y",
#     )
#     audit_dataset = Dataset(
#             data_dict={"train": {"x": audit_data, "y": audit_targets}},
#             default_input="x",
#             default_output="y",
#         )
#     target_info_source = InformationSource(
#         models=[target_model], 
#         datasets=[target_dataset]
#     )

#     reference_info_source = InformationSource(
#         models=[target_model],
#         datasets=[audit_dataset]
#     )
#     metric = PopulationMetric(
#             target_info_source=target_info_source,
#             reference_info_source=reference_info_source,
#             signals=[ModelLoss()],
#             hypothesis_test_func=linear_itp_threshold_func,
#         )
#     audit_obj = Audit(
#         metrics=metric,
#         inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
#         target_info_sources=target_info_source,
#         reference_info_sources=reference_info_source,
#     )
#     audit_obj.prepare()
#     loss_audit_results = audit_obj.run()[0]
 
#     return loss_audit_results[0]