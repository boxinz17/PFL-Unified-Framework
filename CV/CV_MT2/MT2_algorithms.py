import numpy as np
import torch
import torch.nn as nn
import time

def train_LSGD(loss_fn, test_accu_fn, fmj_grads_LSGD, w0, beta0, sync_step, n_commun, devices_train_list, train_loader_list, val_loader_list,
               lambda_global, lambda_penal, repo_step, eta, device, rd_seed=111):
    np.random.seed(rd_seed)
    
    M = len(beta0)
    
    w_list = []
    for m in range(M):
        w_list.append(w0.copy())

    beta = beta0.copy()

    train_loss = []
    test_accu = []
    train_loss.append(loss_fn(w0, beta0, train_loader_list, lambda_global, lambda_penal, device))
    test_accu.append(test_accu_fn(w0, beta0, val_loader_list, device))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(0, 0.0, train_loss[-1], test_accu[-1]))

    start_time = time.time()
    k_commun = 0  # number of communications happend
    k_iter = 0
    while True:
        k_iter += 1
        
        # Synchronization
        if k_iter % sync_step == 0:
            sync_weight = torch.zeros(10, 784).to(device)
            sync_bias = torch.zeros(10).to(device)
            
            # Upload all global parameters to do average
            for m in range(M):
                sync_weight += w_list[m][0]
                sync_bias += w_list[m][1]        
            sync_weight = sync_weight / M
            sync_bias = sync_bias / M

            w_avg = [sync_weight, sync_bias]

            # Download the parameters to each device    
            for m in range(M):
                w_list[m] = w_avg.copy()

            k_commun += 1  # Add number of communications when sync happens
            
            # Compute the loss
            train_loss.append(loss_fn(w_avg, beta, train_loader_list, lambda_global, lambda_penal, device))
            test_accu.append(test_accu_fn(w_avg, beta, val_loader_list, device))
    
            if k_commun % repo_step == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_commun, end_time-start_time, train_loss[-1], test_accu[-1]))
        
        # Choose sample indices
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Compute gradients
        w_grads, beta_grads = fmj_grads_LSGD(w_list, beta, index_choice, devices_train_list, lambda_global, lambda_penal, device)
        
        # Compute new w
        w_new = []
        for m in range(M):
            w_new_m = []
            for i in range(2):
                w_new_m.append(w_list[m][i] - eta * w_grads[m][i])
            w_new.append(w_new_m.copy())
        w_list = w_new.copy()
        
        # Compute new beta
        beta_new = []
        for m in range(M):
            beta_new_m = []
            for i in range(2):
                beta_new_m.append(beta[m][i] - eta * beta_grads[m][i])
            beta_new .append(beta_new_m.copy())
        beta = beta_new.copy()
        
        # check if we should stop
        if k_commun >= n_commun:
            break

    return train_loss, test_accu, w_avg, beta

def train_SCD(loss_fn, test_accu_fn, fmj_grads, w0, beta0, n_commun, devices_train_list, train_loader_list, val_loader_list, \
             lambda_global, lambda_penal, repo_step, eta, p_w, device, rd_seed=111):
    np.random.seed(rd_seed)

    M = len(beta0)
    
    w = w0.copy()
    beta = beta0.copy()

    train_loss = []
    test_accu = []
    train_loss.append(loss_fn(w0, beta0, train_loader_list, lambda_global, lambda_penal, device))
    test_accu.append(test_accu_fn(w0, beta0, val_loader_list, device))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(0, 0.0, train_loss[-1], test_accu[-1]))

    start_time = time.time()
    k_commun = 0  # number of communications happend
    while True:
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Compute gradients
        fmj_grads_result = fmj_grads(w, beta, index_choice, devices_train_list, lambda_global, lambda_penal, device)
        
        # Decide which coordinate to update
        xi = np.random.choice(a=(1,2), p=(p_w, 1-p_w))
        if xi == 1:
            # Update w
            w_new = []
            for i in range(2):
                w_new.append(w[i] - eta * fmj_grads_result[0][i] / p_w)
            w = w_new.copy()

            k_commun += 1  # When update w, it needs to communicate
            
            # Compute the loss
            train_loss.append(loss_fn(w, beta, train_loader_list, lambda_global, lambda_penal, device))
            test_accu.append(test_accu_fn(w, beta, val_loader_list, device))
            if k_commun % repo_step == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_commun, end_time-start_time, train_loss[-1], test_accu[-1]))
        else:
            # Update beta
            beta_new = []
            for m in range(M):
                beta_new_m = []
                for i in range(2):
                    beta_new_m.append(beta[m][i] - eta * fmj_grads_result[1][m][i] / (1-p_w))
                beta_new.append(beta_new_m)
            beta = beta_new.copy()
        
        # check if we should stop
        if k_commun >= n_commun:
            break
            
    return train_loss, test_accu, w, beta

def train_SVRCD(loss_fn, test_accu_fn, F_grads, fmj_grads, w0, beta0, n_commun, devices_train_list, train_loader_list, val_loader_list, lambda_global, lambda_penal, \
                eta, p_w, rho, repo_step, device, rd_seed=111):
    # Initialization
    np.random.seed(rd_seed)

    M = len(beta0)
    p_beta = 1 - p_w

    w_y = w0.copy()
    w_v = w0.copy()
    
    beta_y = beta0.copy()
    beta_v = beta0.copy()
    
    fgrv = F_grads(w_v, beta_v, train_loader_list, lambda_global, lambda_penal, device)
    
    # Begin training
    train_loss = []
    test_accu = []
    train_loss.append(loss_fn(w0, beta0, train_loader_list, lambda_global, lambda_penal, device))
    test_accu.append(test_accu_fn(w0, beta0, val_loader_list, device))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(0, 0.0, train_loss[-1], test_accu[-1]))
    
    start_time = time.time()
    k_commun = 0  # number of communications happend
    k_iter = 0
    while True:
        k_iter += 1
        
        # Choose sample index, zeta and coin
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        # Sample zeta to decide which coordinate to update
        zeta = np.random.choice([1,2], p=(p_w, 1-p_w))
        # Sample coin to decide whether to update (w_v, beta_v)
        coin = np.random.choice([0,1], p=(1-rho, rho))
        
        # Update w_v and beta_v
        if coin == 1:
            beta_v = beta_y.copy()
            w_v = w_y.copy()
            fgrv = F_grads(w_v, beta_v, train_loader_list, lambda_global, lambda_penal, device)
        
        # Compute gradients
        fmj_grads_result_v = fmj_grads(w_v, beta_v, index_choice, devices_train_list, lambda_global, lambda_penal, device)
        fmj_grads_result_y = fmj_grads(w_y, beta_y, index_choice, devices_train_list, lambda_global, lambda_penal, device)

        # Update w's

        ## Compute g_w
        if zeta == 1:
            g_w = []
            for i in range(2):
                g_w.append((fmj_grads_result_y[0][i] - fmj_grads_result_v[0][i]) / p_w + fgrv[0][i])
        else:
            g_w = fgrv[0].copy()

        ## Update w_y
        w_y_new = []
        for i in range(2):
            w_y_new.append(w_y[i] - eta * g_w[i])
        w_y = w_y_new.copy()

        # Update beta's

        ## Compute g_beta
        g_beta = []
        if zeta == 1:
            for m in range(M):
                g_beta.append(fgrv[1][m].copy())
        else:
            for m in range(M):
                g_beta_m = []
                for i in range(2):
                    g_beta_m.append((fmj_grads_result_y[1][m][i] - fmj_grads_result_v[1][m][i]) / p_beta + fgrv[1][m][i])
                g_beta.append(g_beta_m.copy())

        ## Update beta_y
        beta_y_new = []
        for m in range(M):
            beta_y_m = []
            for i in range(2):
                beta_y_m.append(beta_y[m][i] - eta * g_beta[m][i])
            beta_y_new.append(beta_y_m)
        beta_y = beta_y_new.copy()

        # Compute the loss when communication happens
        if zeta == 1:
            k_commun += 1  # communication happens only when zeta=1
            train_loss.append(loss_fn(w_y, beta_y, train_loader_list, lambda_global, lambda_penal, device))
            test_accu.append(test_accu_fn(w_y, beta_y, val_loader_list, device))
            if k_commun % repo_step == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_commun, end_time-start_time, train_loss[-1], test_accu[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break
            
    return train_loss, test_accu, w_y, beta_y

def train_ASVRCD(loss_fn, test_accu_fn, F_grads, fmj_grads, w0, beta0, n_commun, devices_train_list, train_loader_list, val_loader_list, lambda_global, lambda_penal, \
                 eta, p_w, rho, theta1, theta2, nu, gamma, repo_step, device, rd_seed=111):
    # Initialization
    np.random.seed(rd_seed)

    M = len(beta0)
    p_beta = 1 - p_w

    w_y = w0.copy()
    w_z = w0.copy()
    w_v = w0.copy()

    w_x = []
    for i in range(2):
        w_x.append(theta1 * w_z[i] + theta2 * w_v[i] + (1 - theta1 - theta2) * w_y[i])
    
    beta_y = beta0.copy()
    beta_z = beta0.copy()
    beta_v = beta0.copy()

    beta_x = []
    for m in range(M):
        beta_x_m = []
        for i in range(2):
            beta_x_m.append(theta1 * beta_z[m][i] + theta2 * beta_v[m][i] + (1 - theta1 - theta2) * beta_y[m][i])
        beta_x.append(beta_x_m.copy())
    
    fgrv = F_grads(w_v, beta_v, train_loader_list, lambda_global, lambda_penal, device)
    
    # Begin training
    train_loss = []
    test_accu = []
    train_loss.append(loss_fn(w0, beta0, train_loader_list, lambda_global, lambda_penal, device))
    test_accu.append(test_accu_fn(w0, beta0, val_loader_list, device))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(0, 0.0, train_loss[-1], test_accu[-1]))
    
    start_time = time.time()
    k_commun = 0  # number of communications happend
    k_iter = 0
    while True:
        k_iter += 1

        #Update w_x and beta_x
        w_x = []
        for i in range(2):
            w_x.append(theta1 * w_z[i] + theta2 * w_v[i] + (1 - theta1 - theta2) * w_y[i])
        beta_x = []
        for m in range(M):
            beta_x_m = []
            for i in range(2):
                beta_x_m.append(theta1 * beta_z[m][i] + theta2 * beta_v[m][i] + (1 - theta1 - theta2) * beta_y[m][i])
            beta_x.append(beta_x_m.copy())
        
        # Choose sample index, zeta and coin
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        # Sample zeta to decide which coordinate to update
        zeta = np.random.choice([1,2], p=(p_w, 1-p_w))
        # Sample coin to decide whether to update (w_v, beta_v)
        coin = np.random.choice([0,1], p=(1-rho, rho))
        
        # Update w_v and beta_v
        if coin == 1:
            beta_v = beta_y.copy()
            w_v = w_y.copy()
            fgrv = F_grads(w_v, beta_v, train_loader_list, lambda_global, lambda_penal, device)
        
        # Compute gradients
        fmj_grads_result_v = fmj_grads(w_v, beta_v, index_choice, devices_train_list, lambda_global, lambda_penal, device)
        fmj_grads_result_x = fmj_grads(w_x, beta_x, index_choice, devices_train_list, lambda_global, lambda_penal, device)

        # Update w's

        ## Compute g_w
        if zeta == 1:
            g_w = []
            for i in range(2):
                g_w.append((fmj_grads_result_x[0][i] - 0.0*fmj_grads_result_v[0][i]) / p_w + fgrv[0][i])
        else:
            g_w = fgrv[0].copy()

        ## Update w_y
        w_y = []
        for i in range(2):
            w_y.append(w_x[i] - eta * g_w[i])

        ## Update w_z
        w_z_new = []
        for i in range(2):
            w_z_new.append(nu * w_z[i] + (1 - nu) * w_x[i] + (gamma / eta) * (w_y[i] - w_x[i]))
        w_z = w_z_new

        # Update beta's

        ## Compute g_beta
        g_beta = []
        if zeta == 1:
            for m in range(M):
                g_beta.append(fgrv[1][m].copy())
        else:
            for m in range(M):
                g_beta_m = []
                for i in range(2):
                    g_beta_m.append((fmj_grads_result_x[1][m][i] - 0.0*fmj_grads_result_v[1][m][i]) / (p_beta) + fgrv[1][m][i])
                g_beta.append(g_beta_m)

        ## Update beta_y
        beta_y = []
        for m in range(M):
            beta_y_m = []
            for i in range(2):
                beta_y_m.append(beta_x[m][i] - eta * g_beta[m][i])
            beta_y.append(beta_y_m)

        ## Update beta_z
        beta_z_new = []
        for m in range(M):
            beta_z_m = []
            for i in range(2):
                beta_z_m.append(nu * beta_z[m][i] + (1-nu) * beta_x[m][i] + (gamma / eta) * (beta_y[m][i] - beta_x[m][i]))
            beta_z_new.append(beta_z_m)
        beta_z = beta_z_new

        # Compute the loss when communication happens
        if zeta == 1:
            k_commun += 1  # communication happens only when zeta=1
            train_loss.append(loss_fn(w_y, beta_y, train_loader_list, lambda_global, lambda_penal, device))
            test_accu.append(test_accu_fn(w_y, beta_y, val_loader_list, device))
            if k_commun % repo_step == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_commun, end_time-start_time, train_loss[-1], test_accu[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break
            
    return train_loss, test_accu, w_y, beta_y

def train_ASCD(loss_fn, test_accu_fn, fmj_grads, w0, beta0, n_commun, devices_train_list, train_loader_list, val_loader_list, lambda_global, lambda_penal, \
               eta, p_w, theta, nu, gamma, repo_step, device, rd_seed=111):
    # Initialization
    np.random.seed(rd_seed)

    M = len(beta0)
    p_beta = 1 - p_w

    w_y = w0.copy()
    w_z = w0.copy()

    w_x = []
    for i in range(2):
        w_x.append(theta * w_z[i] + (1 - theta) * w_y[i])
    
    beta_y = beta0.copy()
    beta_z = beta0.copy()

    beta_x = []
    for m in range(M):
        beta_x_m = []
        for i in range(2):
            beta_x_m.append(theta * beta_z[m][i] + (1 - theta) * beta_y[m][i])
        beta_x.append(beta_x_m.copy())
    
    # Begin training
    train_loss = []
    test_accu = []
    train_loss.append(loss_fn(w0, beta0, train_loader_list, lambda_global, lambda_penal, device))
    test_accu.append(test_accu_fn(w0, beta0, val_loader_list, device))
    print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(0, 0.0, train_loss[-1], test_accu[-1]))
    
    start_time = time.time()
    k_commun = 0  # number of communications happend
    k_iter = 0
    while True:
        k_iter += 1

        #Update w_x and beta_x
        w_x = []
        for i in range(2):
            w_x.append(theta * w_z[i] + (1 - theta) * w_y[i])
        
        beta_x_new = []
        for m in range(M):
            beta_x_m = []
            for i in range(2):
                #print("m is: {}".format(m))
                #print("length of beta_z[m]: {}".format(len(beta_z[m])))
                #print("length of beta_y[m]: {}".format(len(beta_y[m])))
                beta_x_m.append(theta * beta_z[m][i] + (1 - theta) * beta_y[m][i])
            beta_x_new.append(beta_x_m.copy())
        beta_x = beta_x_new.copy()
        
        # Choose sample index, zeta and coin
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        # Sample zeta to decide which coordinate to update
        xi = np.random.choice([0,1], p=(1-p_w, p_w))

        # Compute gradients
        fmj_grads_result_x = fmj_grads(w_x, beta_x, index_choice, devices_train_list, lambda_global, lambda_penal, device)

        if xi == 0:
            # Update w
            ## Compute g_w
            g_w = []
            for i in range(2):
                g_w.append(fmj_grads_result_x[0][i] / p_w)

            ## Update w_y
            w_y = []
            for i in range(2):
                w_y.append(w_x[i] - eta * g_w[i])

            ## Update w_z
            w_z_new = []
            for i in range(2):
                w_z_new.append(nu * w_z[i] + (1 - nu) * w_x[i] + (gamma / eta) * (w_y[i] - w_x[i]))
            w_z = w_z_new
        else:
            # Update beta
            ## Compute g_beta
            g_beta = []
            for m in range(M):
                g_beta_m = []
                for i in range(2):
                    g_beta_m.append(fmj_grads_result_x[1][m][i] / p_beta)
                g_beta.append(g_beta_m)

            ## Update beta_y
            beta_y_new = []
            for m in range(M):
                beta_y_m = []
                for i in range(2):
                    beta_y_m.append(beta_x[m][i] - eta * g_beta[m][i])
                beta_y_new.append(beta_y_m)
            beta_y = beta_y_new.copy()

            ## Update beta_z
            beta_z_new = []
            for m in range(M):
                beta_z_m = []
                for i in range(2):
                    beta_z_m.append(nu * beta_z[m][i] + (1-nu) * beta_x[m][i] + (gamma / eta) * (beta_y[m][i] - beta_x[m][i]))
                beta_z_new.append(beta_z_m)
            beta_z = beta_z_new.copy()

        # Compute the loss when communication happens
        if xi == 0:
            k_commun += 1  # communication happens only when zeta=1 or coin=1
            train_loss.append(loss_fn(w_y, beta_y, train_loader_list, lambda_global, lambda_penal, device))
            test_accu.append(test_accu_fn(w_y, beta_y, val_loader_list, device))
            if k_commun % repo_step == 0 or k_commun == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_commun, end_time-start_time, train_loss[-1], test_accu[-1]))

        # check if we should stop
        if k_commun >= n_commun:
            break
            
    return train_loss, test_accu, w_y, beta_y