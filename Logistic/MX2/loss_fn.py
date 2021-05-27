import numpy as np
from scipy.stats import logistic

def est_err_MX2(w_true, beta_true_list, w_est, beta_est_list):
    M = len(beta_true_list)
    #err = np.linalg.norm(w_est-w_true)**2 / M
    err = 0.0
    for m in range(M):
        err += np.linalg.norm(beta_est_list[m]-beta_true_list[m])**2
    return err

def loss_fun_MX2(w, beta_list, data_all, lambda_param):
    loss = 0.0
    M = len(beta_list)
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(beta_list[m], axis=1)).squeeze(1))
        # add 1e-12 to avoid numerical error
        loss += (- Y_data * np.log(Y_data_prob + 1e-12) - (1 - Y_data) * np.log(1 - Y_data_prob + 1e-12)).sum() / len(Y_data)
        loss += (lambda_param/2) * (np.linalg.norm(w - beta_list[m]) ** 2)
    loss /= M
    return loss

def loss_fun_MX2_rescale(w, beta_list, data_all, lambda_param):
    loss = 0.0
    M = len(beta_list)
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(beta_list[m], axis=1)).squeeze(1))
        # add 1e-12 to avoid numerical error
        loss += (- Y_data * np.log(Y_data_prob + 1e-12) - (1 - Y_data) * np.log(1 - Y_data_prob + 1e-12)).sum() / len(Y_data)
        loss += (lambda_param/2) * (np.linalg.norm(w/np.sqrt(M) - beta_list[m]) ** 2)
    loss /= M
    return loss

def fmj_grads_MX2(w, beta_list, data_all, lambda_param, index_choice):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        x_data = data_all[m][0][index_choice[m]]
        y_data = data_all[m][1][index_choice[m]]
        y_data_prob = logistic.cdf(np.matmul(x_data, beta_list[m]))
        beta_grad = (y_data_prob - y_data) * x_data
        beta_grad /= M
        beta_grad += (lambda_param / M) * (beta_list[m] - w)
        beta_grads.append(beta_grad)
        w_grad += w - beta_list[m]
    w_grad = (lambda_param / M) * w_grad
    return w_grad, beta_grads

def fmj_grads_MX2_rescale(w, beta_list, data_all, lambda_param, index_choice):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        x_data = data_all[m][0][index_choice[m]]
        y_data = data_all[m][1][index_choice[m]]
        y_data_prob = logistic.cdf(np.matmul(x_data, beta_list[m]))
        beta_grad = (y_data_prob - y_data) * x_data
        beta_grad /= M
        beta_grad += (lambda_param / M) * (beta_list[m] - w / np.sqrt(M))
        beta_grads.append(beta_grad)
        w_grad += w / np.sqrt(M) - beta_list[m]
    w_grad = lambda_param / (M ** 1.5) * w_grad
    return w_grad, beta_grads

def F_grads_MX2_rescale(w, beta_list, data_all, lambda_param):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(beta_list[m], axis=1)).squeeze(1))
        beta_grad = np.matmul(np.transpose(X_data) , np.expand_dims(Y_data_prob-Y_data, axis=1)).squeeze(1) / len(Y_data)
        beta_grad /= M
        beta_grad += (lambda_param / M) * (beta_list[m] - w / np.sqrt(M))
        beta_grads.append(beta_grad)
        w_grad += w / np.sqrt(M) - beta_list[m]
    w_grad = lambda_param / (M ** 1.5) * w_grad
    return w_grad, beta_grads