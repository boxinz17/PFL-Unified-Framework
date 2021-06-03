import numpy as np
from scipy.stats import logistic

def data_generate_fn(n, w_true, beta_true_list):
    M = len(beta_true_list)
    d_g = len(w_true)
    d_l = len(beta_true_list[0])
    data_all = []  # list of data for all devices
    for m in range(M):
        X_data = np.random.uniform(0.0, 0.1, size=(n,d_g+d_l))
        Y_data_prob = logistic.cdf(np.matmul(X_data, np.expand_dims(np.concatenate((w_true, beta_true_list[m])), axis=1)).squeeze(1))
        Y_data = np.random.binomial(1, Y_data_prob, len(Y_data_prob))
        data_all.append([X_data, Y_data])
    return data_all