import os
import warnings
import pickle
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import SubsetRandomSampler

import numpy as np
import flwr as fl

import bts.dataset as dataset
import bts.model as model
import bts.classifier as classifier
import bts.plot as plot
from collections import OrderedDict
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

print('Computation Details')
#print(f'\tDevice Used: ({device})  {torch.cuda.get_device_name(torch.cuda.current_device())}\n')

print('Packages Used Versions:-')
print(f'\tPytorch Version: {torch.__version__}')

# Dataset part used for testing
TEST_SPLIT = 0.2
# Batch size for training. Limited by GPU memory
BATCH_SIZE = 6
# Dataset folder used
DATASET_USED = 'png_dataset'
# Full Dataset path
DATASET_PATH = os.path.join('dataset',DATASET_USED)
#DATASET_PATH = '/Users/siddhikasriram/Documents/Classes-Spring2023/512 OS/osproj/Brain-Tumor-Segmentation/dataset/png_dataset'
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
MODEL_NAME = f"UNet-{FILTER_LIST}-c1.pt"

print(f"Model Name :   {MODEL_NAME}")

def get_indices(length, new=False):
    """ Gets the Training & Testing data indices for a
    paticular "DATASET_USED".Stores the indices and returns
    them back when the same dataset is used.
    Parameters:
        length(int): Length of the dataset used.
        new(bool): Discard the saved indices and get new ones.
    Return:
        train_indices(list): Array of indices used for training purpose.
        test_indices(list): Array of indices used for testing purpose.
    """
    # Pickle file location of the indices.
    file_path = os.path.join('dataset',f'split_indices_{DATASET_USED}.p')
    data = dict()
    if os.path.isfile(file_path) and not new:
        # File found.
        with open(file_path,'rb') as file :
            data = pickle.load(file)
            return data['train_indices'], data['test_indices']
    else:
        # File not found or fresh copy is required.
        indices = list(range(length))
        np.random.shuffle(indices)
        split = int(np.floor(TEST_SPLIT * length))
        train_indices , test_indices = indices[split:], indices[:split]
        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['test_indices'] = test_indices
        with open(file_path,'wb') as file:
            pickle.dump(data,file)
    return train_indices, test_indices

class BrainClient(fl.client.NumPyClient):
    def __init__(self,trainloader,testloader,train_indices,test_indices):
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_indices = train_indices
        self.test_indices = test_indices
    
    def set_parameters(self,parameters):
        unet_model = model.DynamicUNet(FILTER_LIST).to(device)
        unet_classifier = classifier.BrainTumorClassifier(unet_model,device)
        
        params_dict = zip(unet_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        return unet_model,unet_classifier
    
    def get_model_params(self,unet_model):
        return [val.cpu().numpy() for _, val in unet_model.state_dict().items()]

    def fit(self,parameters,config):

        unet_model,unet_classifier = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        
        unet_model.train()
        unet_train_history = unet_classifier.train(epochs,self.trainloader,mini_batch=batch_size)

        unet_model.eval()
        unet_score = unet_classifier.test(self.testloader)

        results = {
            "train_loss": unet_train_history['train_loss'][0],
            "train_accuracy": unet_score,
            "val_loss":unet_train_history['train_loss'][0],
            "val_accuracy":unet_score,
        }
        
        parameters_prime = self.get_model_params(unet_model)
        num_examples_train = len(self.train_indices)
        return parameters_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):
        unet_model,unet_classifier = self.set_parameters(parameters)
        loss = 0
        unet_model.eval()
        unet_score = unet_classifier.test(self.testloader)
        return float(loss), len(self.test_indices), {"accuracy": float(unet_score)}
    
def main() -> None:
    tumor_dataset = dataset.TumorDataset(DATASET_PATH)
    train_indices, test_indices = get_indices(len(tumor_dataset))
    train_indices = [590, 1246, 1166, 744, 516, 949, 40, 685, 2776, 2983, 1693, 802, 2409]
    test_indices = [2234, 2652, 2846, 2067, 2807, 1621, 2945, 1590, 265, 2356, 863]
    train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(tumor_dataset, BATCH_SIZE, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(tumor_dataset, 1, sampler=test_sampler)

    str_ip=input("Enter the server IP:port: ")
    client = BrainClient(trainloader=trainloader, testloader=testloader,train_indices = train_indices,test_indices = test_indices)
    # fl.client.start_numpy_client(server_address="3.21.55.205:8080", root_certificate =Path(Project.pem).read_bytes(), client=client)
    fl.client.start_numpy_client(server_address=str_ip, client=client)
if __name__ == "__main__":
    main()