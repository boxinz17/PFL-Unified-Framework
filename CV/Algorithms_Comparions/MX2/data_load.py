from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import datetime
from torchvision import datasets

def data_prepare(dataset_name, n_devices, n_train, n_val, n_labels=2, batch_size=64, rd_seed=111):
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
    np.random.seed(rd_seed)

    if dataset_name == "MNIST":
        data_path = '../../data/mnist/'
        transform_data = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307),(0.3081))
            ]))
    elif dataset_name == "KMNIST":
        data_path = '../../data/kmnist/'
        transform_data = datasets.KMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1918),(0.3483))
            ]))
    elif dataset_name == "FMNIST":
        data_path = '../../data/fmnist/'
        transform_data = datasets.FashionMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.2861),(0.3530))
            ]))
        
    devices_train_list = []  # list of training data for devices
    train_loader_list = []  # list of train loader for devices
    devices_val_list = []  # list of validation data for devices
    val_loader_list = []  # list of validation loader for devices
    for m in range(n_devices):
        # Choose labels for this device
        labels_choose = list(np.random.choice(10, size=n_labels))
            
        # Choose training data based on chosen labels
        device_data = []
        for img, label in transform_data:
            if label in labels_choose:
                device_data.append((img / torch.norm(img.squeeze(0)).item(), label))
            #if len(device_data) >= n_train + int((1/2) * n_val):
            if len(device_data) >= n_train + n_val:
                break    
        devices_train_list.append(device_data[:n_train])
        train_loader = torch.utils.data.DataLoader(device_data[:n_train], batch_size=batch_size, shuffle=False)
        train_loader_list.append(train_loader)
        # # Choose the other half of validation data
        # # from labels not seen before
        # for img, label in transform_data:
        #     # If the label is not seen, then it has 0.1 prob to
        #     # be included
        #     if label not in labels_choose:
        #         coin = np.random.choice([0,1], p=(0.9, 0.1))
        #         if coin == 1:
        #             device_data.append((img / torch.norm(img.squeeze(0)).item(), label))
        #     if len(device_data) >= n_train + n_val:
        #         break
        devices_val_list.append(device_data[n_train:])
        val_loader = torch.utils.data.DataLoader(device_data[n_train:], batch_size=batch_size, shuffle=False)
        val_loader_list.append(val_loader)
            
    return train_loader_list, devices_train_list, val_loader_list, devices_val_list