from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import datetime
from torchvision import datasets

def data_prepare(dataset_name, n_devices, n_samples, n_labels=2, batch_size=64):
    """
    Return the train_loader and devices_train_list 
    based on the dataset_name.
    """
    # Input: 
    #   dataset_name: A string, should be one of
    #   {"MNIST", "KMNIST", "FMNIST"}
    #   n_devices: number of devices, integer
    #   n_samples: number of samples per device, integer
    #   n_labels: number of labels chosen for each device
    #   batch_size: batch size for creating mini batched
    #               train loader
    # Return:
    #   (train_loader_list, devices_train_list)
    np.random.seed(111)

    if dataset_name == "MNIST":
        data_path = '../data/mnist/'
        transform_data = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307),(0.3081))
            ]))
    elif dataset_name == "KMNIST":
        data_path = '../data/kmnist/'
        transform_data = datasets.KMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1918),(0.3483))
            ]))
    elif dataset_name == "FMNIST":
        data_path = '../data/fmnist/'
        transform_data = datasets.FashionMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.2861),(0.3530))
            ]))
        
    devices_train_list = []  # list of training data for devices
    train_loader_list = []  # list of train loader for devices
    for m in range(n_devices):
        # Choose labels for this device
        labels_choose = list(np.random.choice(10, size=n_labels))
            
         # Choose training data based on chosen labels
        device_train = []
        for img, label in transform_data:
            if label in labels_choose:
                device_train.append((img / torch.norm(img.squeeze(0)).item(), label))
            if len(device_train) >= n_samples:
                break    
        devices_train_list.append(device_train)
            
        train_loader = torch.utils.data.DataLoader(device_train, 
                                                    batch_size=batch_size, shuffle=False)
        train_loader_list.append(train_loader)
            
    return (train_loader_list, devices_train_list)