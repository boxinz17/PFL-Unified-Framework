from matplotlib import pyplot as plt
import numpy as np
import pickle

optim_list = ['LSGD', 'ASCD', 'ASVRCD', 'pFedMe', 'L2SGDplus']

hetero_lv_list = [0.1, 0.3, 1.0]

def avg_result_fun(result_list):
    n_hetro_lv = len(result_list)
    n_repeat = len(result_list[0])
    len_list = len(result_list[0][0])
    result_list_avg = []
    for i in range(n_hetro_lv):
        result_list_avg_sub = np.zeros(len_list)
        for j in range(n_repeat):
            result_list_avg_sub += result_list[i][j]
        result_list_avg_sub /= n_repeat
        result_list_avg.append(result_list_avg_sub)
    return result_list_avg

def std_result_fun(result_list):
    n_hetro_lv = len(result_list)
    n_repeat = len(result_list[0])
    len_list = len(result_list[0][0])
    result_list_std = []
    for i in range(n_hetro_lv):
        result_list_std_sub = []
        for k in range(len_list):
            record_list = []
            for j in range(n_repeat):
                record_list.append(result_list[i][j][k])
            record_list = np.array(record_list)
            result_list_std_sub.append(record_list.std())
        result_list_std_sub = np.array(result_list_std_sub)
        result_list_std.append(result_list_std_sub)
    return result_list_std

loss_results_mean = []
loss_results_std = []
est_results_mean = []
est_results_std = []
for optim in optim_list:
    loss_result = []
    for lv in hetero_lv_list:
        file_name = 'MX2_' + optim + '_train_result_lv' + str(lv) + '.txt'
        result_path = './result/' + file_name
        with open(result_path, "rb") as f:
            loss_result_sub = pickle.load(f)
        loss_result.append(loss_result_sub)
    # Compute the mean
    loss_results_mean.append(avg_result_fun(loss_result))
    # Compute the standard error
    loss_results_std.append(std_result_fun(loss_result))

    est_result = []
    for lv in hetero_lv_list:
        file_name = 'MX2_' + optim + '_est_result_lv' + str(lv) + '.txt'
        result_path = './result/' + file_name
        with open(result_path, "rb") as f:
            loss_est_sub = pickle.load(f)
        est_result.append(loss_est_sub)
    # Compute the mean
    est_results_mean .append(avg_result_fun(est_result))
    # Compute the standard error
    est_results_std .append(std_result_fun(est_result))

for std_times in [0.0, 1.0, 2.0]:
    fig, axes = plt.subplots(2, 3, figsize=[16., 12.])
    for j in range(3):
        # Set y_lim
        top_list = []
        bottom_list = []
        for k in range(len(optim_list)):
            top_list.append(loss_results_mean[k][j].max())
            bottom_list.append(loss_results_mean[k][j].min())
        top_v = max(top_list)
        bottom_v = min(bottom_list)
        shift = (top_v - bottom_v) / 20
        axes[0, j].set_ylim(bottom=bottom_v-shift, top=top_v+shift)

        x = np.linspace(0, len(loss_results_mean[0][j])-1, len(loss_results_mean[0][j]))
        # Plot the mean
        axes[0, j].plot(loss_results_mean[0][j], color='r', linestyle='-', label='LSGD-PFL')
        axes[0, j].plot(loss_results_mean[1][j], color='g', linestyle='-', label='ASCD')
        axes[0, j].plot(loss_results_mean[2][j], color='b', linestyle='-', label='ASVRCD-PFL')
        axes[0, j].plot(loss_results_mean[3][j], color='c', linestyle='-', label='pFedMe')
        axes[0, j].plot(loss_results_mean[4][j], color='m', linestyle='-', label='L2SGD+')
        
        # Plot the error bar with standard error
        axes[0, j].fill_between(x, loss_results_mean[0][j] - std_times * loss_results_std[0][j], 
        loss_results_mean[0][j] + std_times * loss_results_std[0][j], facecolor='r', alpha=0.2,
        edgecolor='r', linestyle='dashdot', label='LSGD-PFL')

        axes[0, j].fill_between(x, loss_results_mean[1][j] - std_times * loss_results_std[1][j], 
        loss_results_mean[1][j] + std_times * loss_results_std[1][j], facecolor='g', alpha=0.2,
        edgecolor='g', linestyle='dashdot', label='ASCD-PFL')

        axes[0, j].fill_between(x, loss_results_mean[2][j] - std_times * loss_results_std[2][j], 
        loss_results_mean[2][j] + std_times * loss_results_std[2][j], facecolor='b', alpha=0.2,
        edgecolor='b', linestyle='dashdot', label='ASVRCD-PFL')

        axes[0, j].fill_between(x, loss_results_mean[3][j] - std_times * loss_results_std[3][j], 
        loss_results_mean[3][j] + std_times * loss_results_std[3][j], facecolor='c', alpha=0.2,
        edgecolor='c', linestyle='dashdot', label='pFedMe')

        axes[0, j].fill_between(x, loss_results_mean[4][j] - std_times * loss_results_std[4][j], 
        loss_results_mean[4][j] + std_times * loss_results_std[4][j], facecolor='m', alpha=0.2,
        edgecolor='m', linestyle='dashdot', label='L2SGD+')
    
        if j == 0:
            axes[0, j].set_ylabel('MX2\nTraining Loss', fontsize=25)

    for j in range(3):
        # Set y_lim
        top_list = []
        bottom_list = []
        for k in range(len(optim_list)):
            top_list.append(est_results_mean[k][j].max())
            bottom_list.append(est_results_mean[k][j].min())
        top_v = max(top_list)
        bottom_v = min(bottom_list)
        shift = (top_v - bottom_v) / 20
        axes[1, j].set_ylim(bottom=bottom_v-shift, top=top_v+shift)

        x = np.linspace(0, len(loss_results_mean[0][j])-1, len(loss_results_mean[0][j]))

        # Plot the mean
        axes[1, j].plot(est_results_mean[0][j], color='r', linestyle='-', label='LSGD-PFL')
        axes[1, j].plot(est_results_mean[1][j], color='g', linestyle='-', label='ASCD-PFL')
        axes[1, j].plot(est_results_mean[2][j], color='b', linestyle='-', label='ASVRCD-PFL')
        axes[1, j].plot(est_results_mean[3][j], color='c', linestyle='-', label='pFedMe')
        axes[1, j].plot(est_results_mean[4][j], color='m', linestyle='-', label='L2SGD+')
    
        # Plot the error bar with standard error
        axes[1, j].fill_between(x, est_results_mean[0][j] - std_times * est_results_std[0][j], 
        est_results_mean[0][j] + std_times * est_results_std[0][j], facecolor='r', alpha=0.2,
        edgecolor='r', linestyle='dashdot', label='LSGD-PFL')

        axes[1, j].fill_between(x, est_results_mean[1][j] - std_times * est_results_std[1][j], 
        est_results_mean[1][j] + std_times * est_results_std[1][j], facecolor='g', alpha=0.2,
        edgecolor='g', linestyle='dashdot', label='ASCD-PFL')

        axes[1, j].fill_between(x, est_results_mean[2][j] - std_times * est_results_std[2][j], 
        est_results_mean[2][j] + std_times * est_results_std[2][j], facecolor='b', alpha=0.2,
        edgecolor='b', linestyle='dashdot', label='ASVRCD-PFL')

        axes[1, j].fill_between(x, est_results_mean[3][j] - std_times * est_results_std[3][j], 
        est_results_mean[3][j] + std_times * est_results_std[3][j], facecolor='c', alpha=0.2,
        edgecolor='c', linestyle='dashdot', label='pFedMe')
    
        axes[1, j].fill_between(x, est_results_mean[4][j] - std_times * est_results_std[4][j], 
        est_results_mean[4][j] + std_times * est_results_std[4][j], facecolor='m', alpha=0.2,
        edgecolor='m', linestyle='dashdot', label='L2SGD+')

        axes[1, j].set_xlabel('Communication Rounds\n' + 'hetero_lv = '+str(hetero_lv_list[j]), fontsize=25)

        if j == 0:
            axes[1, j].set_ylabel('MX2\nEstimation Error', fontsize=25)

    fig.legend(labels=['LSGD-PFL', 'ASCD-PFL', 'ASVRCD-PFL', 'pFedMe', 'L2SGD+'], loc='upper center', ncol=3, fontsize=25)

    plt.savefig('./result/MX2_plot_std' + str(std_times) + '.png', dpi=400, bbox_inches='tight')