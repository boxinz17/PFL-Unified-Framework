import numpy as np
from scipy.stats import logistic
import time
import pickle
from data_generate import data_generate_fn
from loss_fn import est_err_EWS_rescale, loss_fun_EWS_rescale, fmj_grads_EWS_rescale
from algorithms import ASCD_PFL, mu_fun, mathcal_L_fun

np.random.seed(111)

n_repeat = 30  # number of repeat times

d_g = 10  # dimension of global parameter
d_l = 5  # dimension of local parameter
n = 1000  # sample size for each device
M = 20  # number of devices
n_commun = 1000  # number of total communication rounds
lambda_param = 0.0  # parameter for l2-penalty

w_true = np.random.normal(loc=0.0, scale=1.0, size=d_g) # true global parameter

hetero_lv_list = [5.0, 10.0, 15.0]

ASCD_train_result = []
ASCD_est_result = []
ASCD_time = []

start_time_total = time.time()

for hetero_lv in hetero_lv_list:
    ASCD_train_result_sub = []
    ASCD_est_result_sub = []
    ASCD_time_sub = []
    
    np.random.seed(111)
    beta_mu = np.random.normal(loc=0.0, scale=hetero_lv, size=M)  # means of shift for each devices
    beta_true_list = []  # list of true local parameters
    for m in range(M):
        beta_true_list.append(np.random.uniform(beta_mu[m]-0.01, beta_mu[m]+0.01, size=d_l))

    data_all = data_generate_fn(n, w_true, beta_true_list)

    for i_repeat in range(n_repeat):
        ## Parameters of ASCD
        w_y0 = np.zeros(d_g)
        w_z0 = np.zeros(d_g)
        beta_y0_list = []
        for m in range(M):
            beta_y0_list.append(np.zeros(d_l))
        beta_z0_list = []
        for m in range(M):
            beta_z0_list.append(np.zeros(d_l))
    
        mu_prime = mu_fun(w_true, beta_true_list, data_all)
        mathcal_L_w_prime, mathcal_L_beta_prime = mathcal_L_fun(data_all, d_g)

        mu = (mu_prime + lambda_param) / M
        mathcal_L_w = (mathcal_L_w_prime + lambda_param) / M
        mathcal_L_beta = (mathcal_L_beta_prime + lambda_param) / M

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
        loss_train_ASCD, loss_est_ASCD, w_ASCD, beta_list_ASCD = ASCD_PFL(data_all, loss_fun_EWS_rescale, fmj_grads_EWS_rescale, 
                                                                          theta, eta, nu, gamma, p_w, n_commun, est_err_EWS_rescale, 
                                                                          w_true, beta_true_list, w_y0, w_z0, beta_y0_list, 
                                                                          beta_z0_list, lambda_param, repo_period=100)
        end_time = time.time()
        
        ASCD_train_result_sub.append(loss_train_ASCD)
        ASCD_est_result_sub.append(loss_est_ASCD)
        ASCD_time_sub.append(end_time-start_time)

    with open("result/EWS_ASCD_train_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_train_result_sub, f)

    with open("result/EWS_ASCD_est_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_est_result_sub, f)

    with open("result/EWS_ASCD_time_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(ASCD_time_sub, f)

    ASCD_train_result.append(ASCD_train_result_sub)
    ASCD_est_result.append(ASCD_est_result_sub)
    ASCD_time.append(ASCD_time_sub)

end_time_total = time.time()

print("Total time cost: {}(s)".format(end_time_total-start_time_total))

# Save results
with open("result/EWS_ASCD_train_result.txt", "wb") as f:
    pickle.dump(ASCD_train_result, f)

with open("result/EWS_ASCD_est_result.txt", "wb") as f:
    pickle.dump(ASCD_est_result, f)

with open("result/EWS_ASCD_time.txt", "wb") as f:
    pickle.dump(ASCD_time, f)





