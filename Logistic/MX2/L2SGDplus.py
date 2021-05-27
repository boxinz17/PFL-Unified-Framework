import numpy as np
from scipy.stats import logistic
import time

def est_err_MX2_L2SGD(X_true, X_est):
    return np.linalg.norm(X_est - X_true)**2

def loss_fun_MX2_L2SGD(w, X, data_all, lambda_param):
    loss = 0.0
    M = X.shape[0]
    for m in range(M):
        X_data = data_all[m][0]
        Y_data = data_all[m][1]
        Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims( X[m,:], axis=1)).squeeze(1))
        # add 1e-12 to avoid numerical error
        loss += (- Y_data * np.log(Y_data_prob + 1e-12) - (1 - Y_data) * np.log(1 - Y_data_prob + 1e-12)).sum() / len(Y_data)
        loss += (lambda_param/2) * (np.linalg.norm(w -  X[m,:]) ** 2)
    loss /= M
    return loss

def fmj_grads_MX2_L2SGD(w, X, data_all, lambda_param, index_choice):
    M = X.shape[0]
    d = X.shape[1]
    beta_grads = np.zeros((M, d))
    for m in range(M):
        x_data = data_all[m][0][index_choice[m]]
        y_data = data_all[m][1][index_choice[m]]
        y_data_prob = logistic.cdf(np.matmul(x_data, X[m,:]))
        beta_grads[m, :] = (y_data_prob - y_data) * x_data
        beta_grads[m, :] /= M
        beta_grads[m, :] += (lambda_param / M) * (X[m,:] - w)
    return beta_grads

def L2SGD_optimizer(data_all, est_err, loss_fn, fmj_grads, X_true, X0, eta, p_w, lambda_param, n_commun, repo_period=10):
    d = X_true.shape[1]
    M = len(data_all)
    n = len(data_all[0][1])
    X = X0.copy()
    x_bar = np.zeros(d)
    J = np.zeros((M, d, n))
    Psi = np.zeros((M, d))
    
    train_loss = []
    est_loss = []
    train_loss.append(loss_fn(x_bar, X, data_all, lambda_param))
    est_loss.append(est_err(X_true, X))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(0, 0.0, train_loss[-1], est_loss[-1]))

    k_commun = 0
    start_time = time.time()
    while True:
        xi = np.random.choice([0, 1], p=(1-p_w, p_w))

        if xi == 1:
            x_bar = np.mean(X, axis=0)
            for m in range(M):
                g = (lambda_param / (M * p_w)) * (X[m, :] - x_bar) - ((1/ p_w - 1) / M) * Psi[m, :] + \
                    (1 / (M * n)) * np.sum(J[m, :, :], axis=1)
                X[m, :] = X[m, :] - eta * g
                Psi[m, :] = lambda_param * (X[m, :] - x_bar)
            
            # Communication happens when xi=1
            k_commun += 1
            train_loss.append(loss_fn(x_bar, X, data_all, lambda_param))
            est_loss.append(est_err(X_true, X))
            if k_commun % repo_period == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(k_commun, end_time-start_time, train_loss[-1], est_loss[-1]))
        else:
            sample_index = np.random.choice(n, size=n)
            grads_matrix = fmj_grads(x_bar, X, data_all, lambda_param, sample_index)
            for m in range(M):
                g = (1 / (M * (1 - p_w))) * (grads_matrix[m, :] - J[m, :, sample_index[m]]) + \
                    (1 / (M*n)) * np.sum(J[m, :, :], axis=1) + (1 / M) * Psi[m, :]
                X[m, :] = X[m, :] - eta * g
                J[m, :, sample_index[m]] = grads_matrix[m, :].copy()

        # Check if we should stop
        if k_commun >= n_commun:
            break

    return train_loss, est_loss, X

    