from matplotlib import pyplot as plt
import numpy as np
import pickle

obj_list = ['MT2', 'MX2', 'APFL2']
dataset_list = ['MNIST', 'KMNIST', 'FMNIST']

n_labels = 2

loss_results = []
for obj in obj_list:
    loss1 = []
    for dataset in dataset_list:
        loss2 = []
            
        file_name = 'loss_' + obj + '_' + dataset + '_ASCD'
        result_path = '../' + obj + '/result/' + file_name + '.txt'
        with open(result_path, "rb") as f:
            loss = pickle.load(f)
            loss2.append(loss)

        file_name = 'Noreparm_loss_' + obj + '_' + dataset + '_ASCD'
        result_path = '../' + obj + '/result/' + file_name + '.txt'
        with open(result_path, "rb") as f:
            loss = pickle.load(f)
            loss2.append(loss)
            
        loss1.append(loss2)
    loss_results.append(loss1)

test_results = []
for obj in obj_list:
    test1 = []
    for dataset in dataset_list:
        test2 = []
            
        file_name = 'test_' + obj + '_' + dataset + '_ASCD'
        result_path = '../' + obj + '/result/' + file_name + '.txt'
        with open(result_path, "rb") as f:
            test = pickle.load(f)
            test2.append(test)

        file_name = 'Noreparm_test_' + obj + '_' + dataset + '_ASCD'
        result_path = '../' + obj + '/result/' + file_name + '.txt'
        with open(result_path, "rb") as f:
            test = pickle.load(f)
            test2.append(test)
            
        test1.append(test2)
    test_results.append(test1)

fig, axes = plt.subplots(3, 3, figsize=[16., 12.])

for i, obj in enumerate(obj_list):
    for j, dataset in enumerate(dataset_list):
        axes[i, j].plot(loss_results[i][j][0], color='b', linestyle='-', label='Reparam')
        axes[i, j].plot(loss_results[i][j][1], color='r', linestyle='-', label='Non-Reparam')
        if j == 0:
            axes[i, j].set_ylabel(obj + '\nLoss', fontsize=25)
        if i == 2:
            axes[i, j].set_xlabel('Communication Rounds\n' + dataset, fontsize=25)

fig.legend(labels=['Reparam', 'Non-Reparam'], loc='upper center', ncol=3, fontsize=25)

plt.savefig('CV_train_reparam_effect.png', dpi=400, bbox_inches='tight')

fig, axes = plt.subplots(3, 3, figsize=[16., 12.])

for i, obj in enumerate(obj_list):
    for j, dataset in enumerate(dataset_list):
        axes[i, j].plot(test_results[i][j][0], color='b', linestyle='-', label='Reparam')
        axes[i, j].plot(test_results[i][j][1], color='r', linestyle='-', label='Non-Reparam')
        if j == 0:
            axes[i, j].set_ylabel(obj + '\nAccuracy', fontsize=25)
        if i == 2:
            axes[i, j].set_xlabel('Communication Rounds\n' + dataset, fontsize=25)
     
fig.legend(labels=['Reparam', 'Non-Reparam'], loc='upper center', ncol=3, fontsize=25)

plt.savefig('CV_test_reparam_effect.png', dpi=400, bbox_inches='tight')
    