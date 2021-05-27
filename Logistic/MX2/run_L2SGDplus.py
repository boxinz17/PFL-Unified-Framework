import numpy as np
from scipy.stats import logistic
import time
import pickle
from data_generate import data_generate_fn
from algorithms import mu_fun, mathcal_L_fun
from L2SGDplus import est_err_MX2_L2SGD, loss_fun_MX2_L2SGD, fmj_grads_MX2_L2SGD, L2SGD_optimizer

np.random.seed(111)

n_repeat = 30  # number of repeat times

d_gl = 15  # dimension of global parameter & local parameter
n = 1000  # sample size for each device
M = 20  # number of devices
n_commun = 1000  # number of total communication rounds

w_mu = 0.5
w_true = np.random.uniform(w_mu-0.01, w_mu+0.01, size=d_gl)  # true global parameter

hetero_lv_list = [0.1, 0.3, 1.0]

L2SGDplus_train_result = []
L2SGDplus_est_result = []
L2SGDplus_time = []

start_time_total = time.time()

print("n_repeat = {}".format(n_repeat))

for hetero_lv in hetero_lv_list:
    L2SGDplus_train_result_sub = []
    L2SGDplus_est_result_sub = []
    L2SGDplus_time_sub = []

    lambda_param = 1e-2 / hetero_lv  # Penalty parameter

    np.random.seed(111)

    beta_mu = np.random.normal(loc=0.0, scale=hetero_lv, size=M)  # means of shift for each devices
    beta_true_list = []  # list of true local parameters
    for m in range(M):
        shift = np.random.uniform(beta_mu[m]-0.01, beta_mu[m]+0.01, size=d_gl)
        beta_true_list.append(w_true + shift)
    
    data_all = data_generate_fn(n, M, d_gl, beta_true_list)

    X_true = np.zeros((M, d_gl))
    for m in range(M):
        X_true[m, :] = beta_true_list[m].copy()
    
    for i_repeat in range(n_repeat):
        ## Parameters of ASVRCD to deicide eta & p_w
        w_y0 = np.zeros(d_gl)
        w_z0 = np.zeros(d_gl)
        w_v0 = np.zeros(d_gl)
        beta_y0_list = []
        for m in range(M):
            beta_y0_list.append(np.zeros(d_gl))
        beta_z0_list = []
        for m in range(M):
            beta_z0_list.append(np.zeros(d_gl))
        beta_v0_list = []
        for m in range(M):
             beta_v0_list.append(np.zeros(d_gl))
    
        mu_prime = mu_fun(beta_true_list, data_all)
        mathcal_L_prime = mathcal_L_fun(data_all)
    
        mu = mu_prime / (3 * M)
        mathcal_L_w = lambda_param / M
        mathcal_L_beta = (mathcal_L_prime + lambda_param) / M
    
        p_w = mathcal_L_w / (mathcal_L_w + mathcal_L_beta)
        rho = p_w / n
        mathcal_L = 2 * max(mathcal_L_w / p_w, mathcal_L_beta / (1-p_w))
        eta = 1 / (4 * mathcal_L)
        theta2 = 0.5
    
        theta1 = min(0.5, np.sqrt(eta * mu * max(0.5, theta2/rho)))
        gamma = 1 / (max(2 * mu, 4 * theta1 / eta))
        nu = 1 - gamma * mu

        ## Parameters of L2SGDplus
        X0 = np.zeros((M, d_gl))

        # train by L2SGDplus
        if i_repeat == 0:
            print("eta is: {}".format(eta))
            print("p_w is: {}".format(p_w))

        print("hetero_lv: {} | Repeat: {} | Training by L2SGDplus".format(hetero_lv, i_repeat+1))
        
        start_time = time.time()
        loss_train_L2SGDplus, loss_est_L2SGDplus, X_L2SGDplus = L2SGD_optimizer(data_all, est_err_MX2_L2SGD, loss_fun_MX2_L2SGD,fmj_grads_MX2_L2SGD, X_true, X0, eta, p_w, lambda_param, n_commun, repo_period=100)
        end_time = time.time()
        
        L2SGDplus_train_result_sub.append(loss_train_L2SGDplus)
        L2SGDplus_est_result_sub.append(loss_est_L2SGDplus)
        L2SGDplus_time_sub.append(end_time-start_time)

    with open("result/MX2_L2SGDplus_train_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(L2SGDplus_train_result_sub, f)

    with open("result/MX2_L2SGDplus_est_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(L2SGDplus_est_result_sub, f)

    with open("result/MX2_L2SGDplus_time_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(L2SGDplus_time_sub, f)

    L2SGDplus_train_result.append(L2SGDplus_train_result_sub)
    L2SGDplus_est_result.append(L2SGDplus_est_result_sub)
    L2SGDplus_time.append(L2SGDplus_time_sub)

end_time_total = time.time()

print("Total time cost: {}(s)".format(end_time_total-start_time_total))

# Save results
with open("result/MX2_L2SGDplus_train_result.txt", "wb") as f:
    pickle.dump(L2SGDplus_train_result, f)

with open("result/MX2_L2SGDplus_est_result.txt", "wb") as f:
    pickle.dump(L2SGDplus_est_result, f)

with open("result/MX2_L2SGDplus_time.txt", "wb") as f:
    pickle.dump(L2SGDplus_time, f)




