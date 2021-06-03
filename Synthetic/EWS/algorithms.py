import numpy as np
from scipy.stats import logistic
import time

def mu_fun(w, beta_list, data_all):
    M = len(data_all)
    p = len(w) + len(beta_list[0])
    Mat = np.zeros([p,p])
    for m in range(M):
        X_data, Y_data = data_all[m]
        n = len(Y_data)
        Y_data_prob = logistic.cdf(np.matmul(X_data, np.expand_dims(np.concatenate((w, beta_list[m])), axis=1)).squeeze(1))
        Mat += np.matmul(np.matmul(X_data.T, np.diag(Y_data_prob * (1 - Y_data_prob))), X_data) / n
    Mat /= M
    eigen_values, eigen_vectors = np.linalg.eig(Mat)
    return eigen_values[-1]

def mathcal_L_fun(data_all, d_g):
    M = len(data_all)
    max_norm_g = 0.0
    max_norm_l = 0.0
    for m in range(M):
        max_norm_g_m = np.linalg.norm(data_all[m][0][:d_g], axis=1).max()
        max_norm_l_m = np.linalg.norm(data_all[m][0][d_g:], axis=1).max()
        if max_norm_g_m > max_norm_g:
            max_norm_g = max_norm_g_m.copy()
        if max_norm_l_m > max_norm_l:
            max_norm_l = max_norm_l_m.copy()
    return max_norm_g, max_norm_l

def LSGD_PFL(data_all, loss_fn, fmj_grads, eta, n_commun, avg_period, est_err, w_true, beta_true_list,
             w0, beta0_list, lambda_param, repo_period=10):
    M = len(beta0_list)  # number of devices
    n = len(data_all[0][1])  # number of samples for each device

    w_list = []
    for m in range(M):
        w_list.append(w0)
    beta_list = beta0_list

    train_loss = []
    est_loss = []

    train_loss.append(loss_fn(w0, beta0_list, data_all, lambda_param))
    est_loss.append(est_err(w_true, beta_true_list, w0, beta0_list))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(0, 0.0, train_loss[-1], est_loss[-1]))

    k_commun = 0
    k = 0
    start_time = time.time()
    while True:
        k += 1
        sample_index = np.random.choice(n, size=n)

        # Local update
        for m in range(M):
            fmj_grads_result = fmj_grads(w_list[m], beta_list, data_all, lambda_param, sample_index)
            w_list[m] = w_list[m] - eta * fmj_grads_result[0]
            beta_list[m] = beta_list[m] - eta * fmj_grads_result[1][m]
        
        # Average w when communication happens
        if k % avg_period == 0:
            k_commun += 1
            
            w_avg = np.zeros(len(w_list[0]))
            for m in range(M):
                w_avg += w_list[m]
            w_avg /= M
            for m in range(M):
                w_list[m] = w_avg.copy()

            # Compute the loss using w_avg
            train_loss.append(loss_fn(w_avg, beta_list, data_all, lambda_param))
            est_loss.append(est_err(w_true, beta_true_list, w_avg, beta_list))
            
            end_time = time.time()
            if k_commun % repo_period == 0 or k_commun == 1:
                print("commun times {} | time pass : {:.2f}(s) | training loss is {} | estimation error is {}".format(k_commun, end_time-start_time, train_loss[-1], est_loss[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break

    return train_loss, est_loss, w_avg, beta_list

def ASVRCD_PFL(data_all, loss_fn, F_grads, fmj_grads, theta1, theta2, eta, nu, gamma, rho, p_w, n_commun, est_err, w_true, beta_true_list,
               w_y0, w_z0, w_v0, beta_y0_list, beta_z0_list, beta_v0_list, lambda_param, repo_period=10):
    # Initialization    
    M = len(beta_y0_list)  # number of devices
    n = len(data_all[0][1])  # number of samples for each device
    p_beta = 1 - p_w

    w_y = w_y0
    w_z = w_z0
    w_v = w_v0

    w_x = theta1 * w_z + theta2 * w_v + (1 - theta1 - theta2) * w_y

    beta_y_list = beta_y0_list
    beta_z_list = beta_z0_list 
    beta_v_list = beta_v0_list

    beta_x_list = []
    for m in range(M):
        beta_x_list.append(theta1 * beta_z_list[m] + theta2 * beta_v_list[m] + (1 - theta1 - theta2) * beta_y_list[m])

    fgrv = F_grads(w_v, beta_v_list, data_all, lambda_param)
    
    # Begin training
    train_loss = []
    est_loss = []

    train_loss.append(loss_fn(w_y0, beta_y0_list, data_all, lambda_param))
    est_loss.append(est_err(w_true, beta_true_list, w_y0, beta_y0_list))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(0, 0.0, train_loss[-1], est_loss[-1]))
    
    k_commun = 0
    k = 0
    start_time = time.time()
    while True:
        k += 1
        # Update w_x and beta_x_list
        w_x = theta1 * w_z + theta2 * w_v + (1 - theta1 - theta2) * w_y
        for m in range(M):
            beta_x_list[m] = theta1 * beta_z_list[m] + theta2 * beta_v_list[m] + (1 - theta1 - theta2) * beta_y_list[m]
        
        # Choose sample index and zeta
        sample_index = np.random.choice(n, size=n)
        zeta = np.random.choice([1,2], p=(p_w, p_beta))
        coin = np.random.choice([0,1], p=(1-rho, rho))

        ## Update w_v and beta_v_list
        if coin == 1:
            beta_v_list = beta_y_list.copy()
            w_v = w_y.copy()
            fgrv = F_grads(w_v, beta_v_list, data_all, lambda_param)

        # Compute gradients
        fmj_grads_result_v = fmj_grads(w_v, beta_v_list, data_all, lambda_param, sample_index)
        fmj_grads_result_x = fmj_grads(w_x, beta_x_list, data_all, lambda_param, sample_index)

        # Update w's

        ## Compute g_w
        if zeta == 1:
            g_w = (fmj_grads_result_x[0] - fmj_grads_result_v[0]) / p_w + fgrv[0]
        else:
            g_w = fgrv[0]

        ## Update w_y
        w_y = w_x - eta * g_w

        ## Update w_z
        w_z = nu * w_z + (1 - nu) * w_x + (gamma / eta) * (w_y - w_x)

        # Update beta's

        ## Compute g_beta
        g_beta = []
        if zeta == 1:
            for m in range(M):
                g_beta.append(fgrv[1][m])
        else:
            for m in range(M):
                g_beta.append((fmj_grads_result_x[1][m] - fmj_grads_result_v[1][m]) / (p_beta) + fgrv[1][m])

        ## Update beta_y_list
        for m in range(M):
            beta_y_list[m] = beta_x_list[m] - eta * g_beta[m]

        ## Update beta_z_list
        for m in range(M):
            beta_z_list[m] = nu * beta_z_list[m] + (1-nu) * beta_x_list[m] + (gamma / eta) * (beta_y_list[m] - beta_x_list[m])

        # Compute the loss when communication happens
        if zeta == 1 or coin == 1:
            k_commun += 1  # communication happens only when zeta=1 or coin=1
            train_loss.append(loss_fn(w_y, beta_y_list, data_all, lambda_param))
            est_loss.append(est_err(w_true, beta_true_list, w_y, beta_y_list))
            if k_commun % repo_period == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(k_commun, end_time-start_time, 
                                                                                                                  train_loss[-1], est_loss[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break
    
    # Return the result
    return train_loss, est_loss, w_y, beta_y_list

def ASCD_PFL(data_all, loss_fn, fmj_grads, theta, eta, nu, gamma, p_w, n_commun, est_err, w_true, beta_true_list,
             w_y0, w_z0, beta_y0_list, beta_z0_list, lambda_param, repo_period=10):
    # Initialization
    M = len(beta_y0_list)  # number of devices
    n = len(data_all[0][1])  # number of samples for each device
    p_beta = 1 - p_w

    w_y = w_y0
    w_z = w_z0

    w_x = theta * w_z + (1 - theta) * w_y

    beta_y_list = beta_y0_list
    beta_z_list = beta_z0_list

    beta_x_list = []
    for m in range(M):
        beta_x_list.append(theta * beta_z_list[m] + (1 - theta) * beta_y_list[m])
    
    # Begin training
    train_loss = []
    est_loss = []

    train_loss.append(loss_fn(w_y0, beta_y0_list, data_all, lambda_param))
    est_loss.append(est_err(w_true, beta_true_list, w_y0, beta_y0_list))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(0, 0.0, train_loss[-1], est_loss[-1]))
    
    k_commun = 0
    k = 0
    start_time = time.time()
    while True:
        k += 1
        # Update w_x and beta_x_list
        w_x = theta * w_z + (1 - theta) * w_y
        for m in range(M):
            beta_x_list[m] = theta * beta_z_list[m] + (1 - theta) * beta_y_list[m]
        
        # Choose sample index, zeta and coin
        sample_index = np.random.choice(n, size=n)
        xi = np.random.choice([0,1], p=(p_w, 1-p_w))

        # Compute gradients
        fmj_grads_result_x = fmj_grads(w_x, beta_x_list, data_all, lambda_param, sample_index)

        if xi == 0:
            # Update w's

            ## Compute g_w
            g_w = fmj_grads_result_x[0] / p_w

            ## Update w_y
            w_y = w_x - eta * g_w
            
            ## Update w_z
            w_z = nu * w_z + (1 - nu) * w_x + (gamma / eta) * (w_y - w_x)
        else:
            # Update beta's

            ## Compute g_beta
            g_beta = []
            for m in range(M):
                g_beta.append(fmj_grads_result_x[1][m] / p_beta)

            
            ## Update beta_y_list
            for m in range(M):
                beta_y_list[m] = beta_x_list[m] - eta * g_beta[m]

            ## Update beta_z_list
            for m in range(M):
                beta_z_list[m] = nu * beta_z_list[m] + (1-nu) * beta_x_list[m] + (gamma / eta) * (beta_y_list[m] - beta_x_list[m])

        # Compute the loss when communication happens
        if xi == 0:
            k_commun += 1  # communication happens only when zeta=1 or coin=1
            train_loss.append(loss_fn(w_y, beta_y_list, data_all, lambda_param))
            est_loss.append(est_err(w_true, beta_true_list, w_y, beta_y_list))
            if k_commun % repo_period == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | estimation error is: {}".format(k_commun, end_time-start_time, 
                                                                                                                  train_loss[-1], est_loss[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break
    
    # Return the result
    return train_loss, est_loss, w_y, beta_y_list
