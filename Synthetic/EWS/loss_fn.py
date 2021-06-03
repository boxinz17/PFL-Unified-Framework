import numpy as np
from scipy.stats import logistic

def est_err_EWS(w_true, beta_true_list, w_est, beta_est_list):
    M = len(beta_true_list)
    err = np.linalg.norm(w_est-w_true)**2
    for m in range(M):
        err += np.linalg.norm(beta_est_list[m]-beta_true_list[m]) ** 2 / M
    return err

def est_err_EWS_rescale(w_true, beta_true_list, w_est, beta_est_list):
    M = len(beta_true_list)
    err = np.linalg.norm(w_est / np.sqrt(M) -w_true)**2
    for m in range(M):
        err += np.linalg.norm(beta_est_list[m]-beta_true_list[m]) ** 2 / M
    return err

def loss_fun_EWS(w, beta_list, data_all, lambda_param):
    loss = 0.0
    M = len(beta_list)
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(np.concatenate((w, beta_list[m])), axis=1)).squeeze(1))
        # add 1e-12 to avoid numerical error
        loss += (- Y_data * np.log(Y_data_prob + 1e-12) - (1 - Y_data) * np.log(1 - Y_data_prob + 1e-12)).sum() / len(Y_data)
    loss /= M
    loss += (lambda_param / 2) * (np.linalg.norm(w) ** 2)
    for m in range(M):
        loss += (lambda_param / 2) * (np.linalg.norm(beta_list[m]) ** 2)
    return loss

def loss_fun_EWS_rescale(w, beta_list, data_all, lambda_param):
    loss = 0.0
    M = len(beta_list)
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(np.concatenate((w/np.sqrt(M), beta_list[m])), axis=1)).squeeze(1))
        # add 1e-12 to avoid numerical error
        loss += (- Y_data * np.log(Y_data_prob + 1e-12) - (1 - Y_data) * np.log(1 - Y_data_prob + 1e-12)).sum() / len(Y_data)
    loss /= M
    loss += (lambda_param / (2 * M)) * (np.linalg.norm(w) ** 2)
    for m in range(M):
        loss += (lambda_param / 2) * (np.linalg.norm(beta_list[m]) ** 2)
    return loss

def F_grads_EWS_rescale(w, beta_list, data_all, lambda_param):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(np.concatenate((w / np.sqrt(M), beta_list[m])), axis=1)).squeeze(1))
        grad = np.matmul(np.transpose(X_data) , np.expand_dims(Y_data_prob-Y_data, axis=1)).squeeze(1) / len(Y_data)
        w_grad += grad[:len(w)] / np.sqrt(M)
        beta_grads.append(grad[len(w):] + lambda_param * beta_list[m])
    w_grad /= M
    w_grad += (lambda_param / M) * w
    return w_grad, beta_grads

def fmj_grads_EWS(w, beta_list, data_all, lambda_param, index_choice):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        x_data = data_all[m][0][index_choice[m]]
        y_data = data_all[m][1][index_choice[m]]
        y_data_prob = logistic.cdf(np.matmul(x_data, np.concatenate((w, beta_list[m]))))
        grad = (y_data_prob - y_data) * x_data
        w_grad += grad[:len(w)]
        beta_grads.append(grad[len(w):] + lambda_param * beta_list[m])
    w_grad /= M
    w_grad += lambda_param * w
    return w_grad, beta_grads

def fmj_grads_EWS_rescale(w, beta_list, data_all, lambda_param, index_choice):
    M = len(beta_list)
    w_grad = np.zeros(len(w))
    beta_grads = []
    for m in range(M):
        x_data = data_all[m][0][index_choice[m]]
        y_data = data_all[m][1][index_choice[m]]
        y_data_prob = logistic.cdf(np.matmul(x_data, np.concatenate((w / np.sqrt(M), beta_list[m]))))
        grad = (y_data_prob - y_data) * x_data
        w_grad += grad[:len(w)] / np.sqrt(M)
        beta_grads.append(grad[len(w):] + lambda_param * beta_list[m])
    w_grad /= M
    w_grad += (lambda_param / M) * w
    return w_grad, beta_grads



