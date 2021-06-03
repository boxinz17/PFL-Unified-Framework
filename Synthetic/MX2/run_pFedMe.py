import numpy as np
from scipy.stats import logistic
import time
import pickle
from data_generate import data_generate_fn
from loss_fn import est_err_MX2, loss_fun_MX2
from pFedMe import theta_tilde_grad, theta_tilde_solver, pFedMe_optimizer

np.random.seed(111)

n_repeat = 30  # number of repeat times

d_gl = 15  # dimension of global parameter & local parameter
n = 1000  # sample size for each device
M = 20  # number of devices
n_commun = 1000  # number of total communication rounds

w_mu = 0.5
w_true = np.random.uniform(w_mu-0.01, w_mu+0.01, size=d_gl)  # true global parameter

hetero_lv_list = [0.3, 1.0]

pFedMe_train_result = []
pFedMe_est_result = []
pFedMe_time = []

start_time_total = time.time()

print("n_repeat = {}".format(n_repeat))

for hetero_lv in hetero_lv_list:
    pFedMe_train_result_sub = []
    pFedMe_est_result_sub = []
    pFedMe_time_sub = []

    lambda_param = 1e-2 / hetero_lv  # Penalty parameter

    np.random.seed(111)

    beta_mu = np.random.normal(loc=0.0, scale=hetero_lv, size=M)  # means of shift for each devices
    beta_true_list = []  # list of true local parameters
    for m in range(M):
        shift = np.random.uniform(beta_mu[m]-0.01, beta_mu[m]+0.01, size=d_gl)
        beta_true_list.append(w_true + shift)
    
    data_all = data_generate_fn(n, M, d_gl, beta_true_list)
    
    for i_repeat in range(n_repeat):
        # Parameters of pFedMe
        w0 = np.zeros(d_gl)
        theta0_list = []
        for m in range(M):
            theta0_list.append(np.zeros(d_gl))
        eta_pFedMe = 0.005

        # train by pFedMe
        start_time = time.time()
        print("hetero_lv: {} | Repeat: {} | Training by pFedMe".format(hetero_lv, i_repeat+1)) 
        loss_train_LSGD, loss_est_LSGD, w_LSGD, beta_list_LSGD = pFedMe_optimizer(data_all, loss_fun_MX2, est_err_MX2, w_true, 
                                                                                  beta_true_list, lambda_param, eta_pFedMe, w0, 
                                                                                  theta0_list, n_commun, repo_period=100)
        end_time = time.time()
        
        pFedMe_train_result_sub.append(loss_train_LSGD)
        pFedMe_est_result_sub.append(loss_est_LSGD)
        pFedMe_time_sub.append(end_time-start_time)
    
    with open("result/MX2_pFedMe_train_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(pFedMe_train_result_sub, f)

    with open("result/MX2_pFedMe_est_result_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(pFedMe_est_result_sub, f)

    with open("result/MX2_pFedMe_time_lv" + str(hetero_lv) + ".txt", "wb") as f:
        pickle.dump(pFedMe_time_sub, f)

    pFedMe_train_result.append(pFedMe_train_result_sub)
    pFedMe_est_result.append(pFedMe_est_result_sub)
    pFedMe_time.append(pFedMe_time_sub)

end_time_total = time.time()

print("Total time cost: {}(s)".format(end_time_total-start_time_total))

# Save results
with open("result/MX2_pFedMe_train_result.txt", "wb") as f:
    pickle.dump(pFedMe_train_result, f)

with open("result/MX2_pFedMe_est_result.txt", "wb") as f:
    pickle.dump(pFedMe_est_result, f)

with open("result/MX2_pFedMe_time.txt", "wb") as f:
    pickle.dump(pFedMe_time, f)