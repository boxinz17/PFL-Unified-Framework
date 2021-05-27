import numpy as np
from scipy.stats import logistic

def data_generate_fn(n, M, d_gl, beta_true_list):
    data_all = []  # list of data for all devices
    for m in range(M):
        X_data = np.random.uniform(0.2, 0.5, size=(n, d_gl))
        Y_data_prob = logistic.cdf(np.matmul(X_data, np.expand_dims(beta_true_list[m], axis=1)).squeeze(1))
        Y_data = np.random.binomial(1, Y_data_prob, len(Y_data_prob))
        data_all.append([X_data, Y_data])
    return data_all