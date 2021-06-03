from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time
import pickle

import data_load
from MX2_loss import loss_fn, test_accu_fn, F_grads, fmj_grads, fmj_grads_LSGD
from MX2_algorithms import train_LSGD, train_ASVRCD, train_ASCD

start_time = time.time()

rd_seed = 111

# Setup device
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

# Initialize parameters
n_commun = 1000
repo_step = 100
sync_step = 5

M = 20
n_train = 100
n_val = 300

obj = 'MX2'
data_name = 'FMNIST'

n_labels_list = [2, 4, 8]

for n_labels in n_labels_list:
    print("n_labels: {}".format(n_labels))
    print("\n")

    lambda_penal = 1.0 / n_labels

    # Load the data
    train_loader_list, devices_train_list, val_loader_list, devices_val_list = data_load.data_prepare(data_name, M, n_train, n_val, 
    n_labels, batch_size=64, rd_seed=rd_seed)

    # Initialize parameters for ASVRCD-PFL & ASCD-PFL
    mathcal_L_prime = 1.0
    mu_prime = 1e-2
    
    mu = mu_prime / (3 * M)

    mathcal_L_w = lambda_penal / M
    mathcal_L_beta = (mathcal_L_prime + lambda_penal) / M

    p_w = mathcal_L_w / (mathcal_L_beta + mathcal_L_w)

    rho = p_w / n_train
    mathcal_L = 2 * max(mathcal_L_w / p_w, mathcal_L_beta / (1-p_w))
    eta = 1 / (4 * mathcal_L)
    theta2 = 0.5
    theta1 = min(0.5, np.sqrt(eta * mu * max(0.5, theta2/rho)))
    gamma = 1 / (max(2 * mu, 4 * theta1 / eta))
    nu = 1 - gamma * mu
    
    # Print useful infomration
    print("Useful Information:")
    print("lambda_penal: {}".format(lambda_penal))
    print("n_commun: {}".format(n_commun))
    print("sync_step: {}".format(sync_step))
    print("M: {}".format(M))
    print("n_train: {}".format(n_train))
    print("n_val: {}".format(n_val))
    print("mu: {}".format(mu))
    print("p_w: {}".format(p_w))
    print("eta: {}".format(eta))
    print("rho: {}".format(rho))
    print("theta1: {}".format(theta1))
    print("theta2: {}".format(theta2))
    print("gamma: {}".format(gamma))
    print("nu: {}".format(nu))
    print("rho: {}".format(rho))
    print("sync_step: {}".format(sync_step))
    print("\n")

    # Train by LSGD-PFL
    w0 = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    beta0 = []
    for m in range(M):
        beta0.append([torch.zeros(10, 784).to(device), torch.zeros(10).to(device)])
    eta_LSGD = 0.01
    print("Training by LSGD-PFL")
    print("eta_lSGD: {}".format(eta_LSGD))
    loss_LSGD, test_LSGD, _, _ = train_LSGD(loss_fn, test_accu_fn, fmj_grads_LSGD, w0, beta0, sync_step, n_commun, devices_train_list, 
    train_loader_list, val_loader_list, lambda_penal, repo_step, eta, device, rd_seed=rd_seed)

    # Train by ASCD-PFL
    theta = min(1 / eta, 0.8)
    w0 = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    beta0 = []
    for m in range(M):
        beta0.append([torch.zeros(10, 784).to(device), torch.zeros(10).to(device)])
    print("Training by ASCD-PFL")
    print("theta: {}".format(theta))
    loss_ASCD, test_ASCD, _, _ = train_ASCD(loss_fn, test_accu_fn, fmj_grads, w0, beta0, n_commun, devices_train_list, 
    train_loader_list, val_loader_list, lambda_penal, eta, p_w, theta, nu, gamma, repo_step, device, rd_seed=rd_seed)

    # Train by ASVRCD-PFL
    w0 = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    beta0 = []
    for m in range(M):
        beta0.append([torch.zeros(10, 784).to(device), torch.zeros(10).to(device)])
    print("Training by ASVRCD-PFL")
    loss_ASVRCD, test_ASVRCD, _, _ = train_ASVRCD(loss_fn, test_accu_fn, F_grads, fmj_grads, w0, beta0, n_commun, devices_train_list, 
    train_loader_list, val_loader_list, lambda_penal, eta, p_w, rho, theta1, theta2, nu, gamma, repo_step, device, rd_seed=rd_seed)

    #Save the result
    with open("./result_nlabel" + str(n_labels) + "/loss_" + obj + "_" + data_name + "_LSGD.txt", "wb") as f:   #Pickling
        pickle.dump(loss_LSGD, f)
    with open("./result_nlabel" + str(n_labels) + "/test_" + obj + "_" + data_name + "_LSGD.txt", "wb") as f:   #Pickling
        pickle.dump(test_LSGD, f)

    with open("./result_nlabel" + str(n_labels) + "/loss_" + obj + "_" + data_name + "_ASCD.txt", "wb") as f:   #Pickling
        pickle.dump(loss_ASCD, f)
    with open("./result_nlabel" + str(n_labels) + "/test_" + obj + "_" + data_name + "_ASCD.txt", "wb") as f:   #Pickling
        pickle.dump(test_ASCD, f)

    with open("./result_nlabel" + str(n_labels) + "/loss_" + obj + "_" + data_name + "_ASVRCD.txt", "wb") as f:   #Pickling
        pickle.dump(loss_ASVRCD, f)
    with open("./result_nlabel" + str(n_labels) + "/test_" + obj + "_" + data_name + "_ASVRCD.txt", "wb") as f:   #Pickling
        pickle.dump(test_ASVRCD, f)


end_time = time.time()
print("Total time cost: {}".format(end_time-start_time))