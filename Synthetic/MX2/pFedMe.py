import numpy as np
from scipy.stats import logistic
import time

def theta_tilde_grad(theta, data, lambda_param, K, w):
    X_data = data[0]
    Y_data = data[1]
    Y_data_prob = logistic.cdf(np.matmul(X_data , np.expand_dims(theta, axis=1)).squeeze(1))
    theta_grad = np.matmul(np.transpose(X_data) , np.expand_dims(Y_data_prob-Y_data, axis=1)).squeeze(1) / len(Y_data)
    theta_grad += lambda_param * (theta - w)
    return theta_grad

def theta_tilde_solver(data, lambda_param, K, w, theta0):
    theta = theta0.copy()
    for k in range(K):
        theta_grad = theta_tilde_grad(theta, data, lambda_param, K, w)
        theta -= theta_grad
    return theta

def pFedMe_optimizer(data_all, loss_fn, est_err, w_true, beta_true_list, lambda_param, eta, w0, theta0_list, n_commun, repo_period=10, beta_param=1, R=20, K=5, batch_size=20):
    M = len(theta0_list)
    w_avg = w0.copy()
    w_list = []
    for m in range(M):
        w_list.append(w0.copy())
    theta_list = theta0_list.copy()

    train_loss = []  # training loss
    est_loss = []  # estimation loss

    train_loss.append(loss_fn(w0, theta0_list, data_all, lambda_param))
    est_loss.append(est_err(w_true, beta_true_list, w0, theta0_list))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(0, 0.0, train_loss[-1], est_loss[-1]))

    k_commun = 0
    start_time = time.time()
    while True:
        
        # Each machine get w_global and do local computation
        for m in range(M):
            n = len(data_all[m][1])
            w_list[m] = w_avg.copy()
            # Local update rounds
            for r in range(R):
                sample_indices = set(np.random.choice(n, size=batch_size))
                X_subset = [data_all[m][0][i] for i in sample_indices]
                Y_subset = [data_all[m][1][i] for i in sample_indices]
                theta_list[m] = theta_tilde_solver((X_subset, Y_subset), lambda_param, K, w_list[m], theta_list[m])
                w_list[m] -= eta * lambda_param * (w_list[m] - theta_list[m])

        # Communication happens
        k_commun += 1
        w_avg = np.zeros(len(w_avg))
        for m in range(M):
            w_avg += w_list[m]
        w_avg /= M

        # Compute the loss using w_avg
        train_loss.append(loss_fn(w_avg, theta_list, data_all, lambda_param))
        est_loss.append(est_err(w_true, beta_true_list, w_avg, theta_list))

        if k_commun % repo_period == 0 or k_commun == 1:
            end_time = time.time()
            print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(k_commun, end_time-start_time, 
                                                                                                                  train_loss[-1], est_loss[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break

    return train_loss, est_loss, w_avg, theta_list