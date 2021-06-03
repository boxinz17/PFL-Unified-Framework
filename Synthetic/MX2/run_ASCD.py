import numpy as np
from scipy.stats import logistic
import time
import pickle
from data_generate import data_generate_fn
from loss_fn import est_err_MX2, loss_fun_MX2_rescale, fmj_grads_MX2_rescale
from algorithms import ASCD_PFL, mu_fun, mathcal_L_fun

np.random.seed(111)

n_repeat = 30  # number of repeat times

d_gl = 15  # dimension of global parameter & local parameter
n = 1000  # sample size for each device
M = 20  # number of devices
n_commun = 1000  # number of total communication rounds

w_mu = 0.5
w_true = np.random.uniform(w_mu-0.01, w_mu+0.01, size=d_gl)  # true global parameter

hetero_lv_list = [0.1, 0.3, 1.0]

ASCD_train_result = []
ASCD_est_result = []
ASCD_time = []

start_time_total = time.time()

print("n_repeat = {}".format(n_repeat))

for hetero_lv in hetero_lv_list:
    ASCD_train_result_sub = []
    ASCD_est_result_sub = []
    ASCD_time_sub = []

    lambda_param = 1e-2 / hetero_lv  # Penalty parameter

    np.random.seed(111)

    beta_mu = np.random.normal(loc=0.0, scale=hetero_lv, size=M)  # means of shift for each devices
    beta_true_list = []  # list of true local parameters
    for m in range(M):
        shift = np.random.uniform(beta_mu[m]-0.01, beta_mu[m]+0.01, size=d_gl)
        beta_true_list.append(w_true + shift)
    
    data_all = data_generate_fn(n, M, d_gl, beta_true_list)
    
    for i_repeat in range(n_repeat):
        ## Parameters of ASCD
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

        theta = min(1 / eta, 0.8)

        # train by ASCD
        print("hetero_lv: {} | Repeat: {} | Training by ASCD_PFL".format(hetero_lv, i_repeat+1))
        start_time = time.time()
        loss_train_ASCD, loss_est_ASCD, w_ASCD, beta_list_ASCD = ASCD_PFL(data_all, loss_fun_MX2_rescale, fmj_grads_MX2_rescale, 
                                                                          theta, eta, nu, gamma, p_w, n_commun, est_err_MX2, 
                                                                          w_true, beta_true_list,  w_y0, w_z0, beta_y0_list, 
                                                                          beta_z0_list, lambda_param, repo_period=100)
        end_time = time.time()
        
        ASCD_train_result_sub.append(loss_train_ASCD)
        ASCD_est_result_sub.append(loss_est_ASCD)
        ASCD_time_sub.append(end_time-start_time)

    with open("result/MX2_ASCD_train_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_train_result_sub, f)

    with open("result/MX2_ASCD_est_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_est_result_sub, f)

    with open("result/MX2_ASCD_time_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_time_sub, f)

    ASCD_train_result.append(ASCD_train_result_sub)
    ASCD_est_result.append(ASCD_est_result_sub)
    ASCD_time.append(ASCD_time_sub)

end_time_total = time.time()

print("Total time cost: {}(s)".format(end_time_total-start_time_total))

# Save results
with open("result/MX2_ASCD_train_result.txt", "wb") as f:
    pickle.dump(ASCD_train_result, f)

with open("result/MX2_ASCD_est_result.txt", "wb") as f:
    pickle.dump(ASCD_est_result, f)

with open("result/MX2_ASCD_time.txt", "wb") as f:
    pickle.dump(ASCD_time, f)




