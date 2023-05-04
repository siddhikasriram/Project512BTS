from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch
import warnings
import os
import pickle
import numpy as np
from torch.utils.data import SubsetRandomSampler
import bts.dataset as dataset
import bts.model as model
import bts.classifier as classifier
import bts.plot as plot
from collections import OrderedDict

# Dataset part used for testing
TEST_SPLIT = 0.2
# Batch size for training. Limited by GPU memory
BATCH_SIZE = 6
# Dataset folder used
DATASET_USED = 'png_dataset'
# Full Dataset path
DATASET_PATH = os.path.join('dataset',DATASET_USED)
# Training Epochs
EPOCHS = 1
# Filters used in UNet Model
FILTER_LIST = [16,32,64,128,256]
# Flag to train the model
TRAIN = True
# Flag to load saved model
LOAD_MODEL = False
# Flag to save model trained
SAVE_MODEL = True
# Model name to save or load.
MODEL_NAME = f"UNet-{FILTER_LIST}-s1.pt"

warnings.filterwarnings("ignore")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 6,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(unet_model,unet_classifier):
    """Return an evaluation function for server-side evaluation."""

    tumor_dataset = dataset.TumorDataset(DATASET_PATH)
    train_indices = [590, 1246, 1166, 744, 516, 949, 40, 685, 2776, 2983, 1693, 802, 2409]
    test_indices = [2234, 2652, 2846, 2067, 2807, 1621, 2945, 1590, 265, 2356, 863]
    train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    # trainloader = torch.utils.data.DataLoader(tumor_dataset, BATCH_SIZE, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(tumor_dataset, 1, sampler=test_sampler)
    valloader = testloader
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters

        params_dict = zip(unet_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        unet_model.load_state_dict(state_dict, strict=True)

        unet_model.eval()
        unet_score = unet_classifier.test(valloader)
        loss = 0
        return loss, {"accuracy": unet_score}

    return evaluate

def main():

    unet_model = model.DynamicUNet(FILTER_LIST).to(device)
    unet_classifier = classifier.BrainTumorClassifier(unet_model,device)
    str_ip=input("Enter IP address and port number: ")
    min_clients=int(input("Enter the minimum number of clients participating: "))
    model_parameters = [val.cpu().numpy() for _, val in unet_model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=get_evaluate_fn(unet_model=unet_model,unet_classifier=unet_classifier),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

     # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address=str_ip,
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
