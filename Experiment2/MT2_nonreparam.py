from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import datetime

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

def loss_fn(w, beta_list, train_loader_list, lambda_global, lambda_penal):
    # Define loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')
    
    M = len(beta_list)

    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0]
        global_model.bias[:] = w[1]
    
    loss_train = 0.0
    for m in range(M):
        loss_train_local = 0
        
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up training dataset
        train_loader = train_loader_list[m]
        
        # Begin computing the loss
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            
            # Compute loss of global model
            with torch.no_grad():
                outs_global = global_model(imgs.view(imgs.shape[0], -1))
                loss_global = loss_CE_fn(outs_global, labels)
             
            # Compute loss of local model
            with torch.no_grad():
                outs_local = local_model(imgs.view(imgs.shape[0], -1))
                loss_local = loss_CE_fn(outs_local, labels)
                #loss_local += 
                
            # Compute loss of penalty term
            with torch.no_grad():
                p_local = torch.nn.utils.parameters_to_vector(local_model.parameters())
                p_global = torch.nn.utils.parameters_to_vector(global_model.parameters())
                loss_penalty = penalty_fn(p_local, p_global)
                
            # Add all three terms
            with torch.no_grad():
                loss = lambda_global * loss_global + loss_local + (lambda_penal/2) * loss_penalty
            
            # Update local loss_train
            loss_train_local += loss.item()
        
        # Compute average local loss train
        loss_train_local = loss_train_local / len(train_loader)
        
        # Update loss train
        loss_train += loss_train_local
    
    # Compute average loss train
    loss_train = loss_train / M
    
    return loss_train

def F_grads(w, beta_list, train_loader_list, lambda_global, lambda_penal):
    # Define loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(beta_list)
    
    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0]
        global_model.bias[:] = w[1]
    
    # Set up local model
    local_models_list = []
    for m in range(M):
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        local_models_list.append(local_model)
    
    loss = 0.0
    for m in range(M):
        # Set up local trianing data
        train_loader = train_loader_list[m]
        
        # Extract local model from list
        local_model = local_models_list[m]
        
        loss_device = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            
            # Compute loss of global model
            outs_global = global_model(imgs.view(imgs.shape[0], -1))
            loss_global = loss_CE_fn(outs_global, labels)
             
            # Compute loss of local model
            outs_local = local_model(imgs.view(imgs.shape[0], -1))
            loss_local = loss_CE_fn(outs_local, labels)
                
            # Compute loss of penalty term
            p_local = torch.nn.utils.parameters_to_vector(local_model.parameters())
            p_global = torch.nn.utils.parameters_to_vector(global_model.parameters())
            loss_penalty = penalty_fn(p_local, p_global)
            
            # Add all three terms
            loss_device += lambda_global * loss_global + loss_local + (lambda_penal/2) * loss_penalty
            
        # Average Local Loss
        loss_device = loss_device / len(train_loader)
        
        # Update overall loss
        loss += loss_device
    
    # Average overall loss
    loss = loss / M
    
    # Compute gradients
    loss.backward()
    
    # Store the gradients w.r.t w
    w_grad = []
    for param in global_model.parameters():
        w_grad.append(param.grad)
        
    # Store the gradients w.r.t. beta_m's
    beta_grads = []
    for m in range(M):
        local_model = local_models_list[m]
        beta_grad = []
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grad, beta_grads]

def fmj_grads(w, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal):
    # Return: A list of grads w.r.t. beta_m's
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(beta_list)
    
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0]
        global_model.bias[:] = w[1]
        
    local_models_list = []
    for m in range(M):
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        local_models_list.append(local_model)
    
    loss = 0.0
    for m in range(M):
        train_device = devices_train_list[m]
        img, label = train_device[index_choice[m]]
        local_model = local_models_list[m]
        
        img = img.to(device=device)
        label = torch.Tensor([label]).to(device=device).long()
        
        # Compute loss of global model
        out_global = global_model(img.view(1,-1))
        loss_global = loss_CE_fn(out_global, label)
        
        # Compute loss of local model
        out_local = local_model(img.view(1,-1))
        loss_local = loss_CE_fn(out_local, label)
        
        # Compute loss of penalty term
        p_local = torch.nn.utils.parameters_to_vector(local_model.parameters())
        p_global = torch.nn.utils.parameters_to_vector(global_model.parameters())
        loss_penalty = penalty_fn(p_local, p_global)
        
        # Add all three terms
        loss += lambda_global * loss_global + loss_local + (lambda_penal/2) * loss_penalty
    
    loss = loss / M
    
    loss.backward()
    
    w_grad = []
    for param in global_model.parameters():
        w_grad.append(param.grad)
    
    beta_grads = []
    for m in range(M):
        beta_grad = []
        local_model = local_models_list[m]
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grad, beta_grads]

def train_GD(w0, beta0, n_epochs, train_loader_list, lambda_global, \
             lambda_penal, repo_step, eta, obj, data_name):
    loss_GD = []  # A list recording the loss values
    
    w_curr = w0
    beta_curr = beta0
    
    M = len(beta0)

    start_time = datetime.datetime.now()
    for epoch in range(1, n_epochs + 1):
        # Compute gradients
        F_grads_result = F_grads(w_curr, beta_curr, train_loader_list, 
                         lambda_global, lambda_penal)
        F_w_grad_result = F_grads_result[0]
        fm_beta_grads_result = F_grads_result[1]
        
        # Update w
        w_new = []
        for i in range(2):
            w_new.append(w_curr[i] - eta * F_w_grad_result[i])
    
        # Update beta
        beta_new = []
        for m in range(M):
            beta_new_m = []
            for i in range(2):
                beta_new_m.append(beta_curr[m][i] - eta * fm_beta_grads_result[m][i])
            beta_new.append(beta_new_m)
            
        # Update parameters
        w_curr = w_new
        beta_curr = beta_new
        
        # Compute the loss
        loss = loss_fn(w_curr, beta_curr, train_loader_list, 
                       lambda_global, lambda_penal)
        loss_GD.append(loss)
        
        if epoch == 1 or epoch % repo_step == 0:
            now_time = datetime.datetime.now()
            pass_time = (now_time - start_time).seconds
            print("epoch: {}, loss: {:.10f}, time pass: {}s | GD {} {} Non-Reparam".format(epoch, loss, pass_time, obj, data_name))
            
    return loss_GD, w_curr, beta_curr

def train_SGD(w0, beta0, n_epochs, devices_train_list, train_loader_list, \
              lambda_global, lambda_penal, repo_step, eta, obj, data_name, rd_seed=111):
    np.random.seed(rd_seed)
    
    loss_SGD = []
    
    w_curr = w0
    beta_curr = beta0

    M = len(beta0)
    
    start_time = datetime.datetime.now()
    for epoch in range(1, n_epochs + 1):
        # Sample j's
        index_choice = []
        for m in range(M):
             index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Compute gradients
        avg_fmj_grads_result = fmj_grads(w_curr, beta_curr, 
                                         index_choice, devices_train_list, 
                                         lambda_global, lambda_penal)
        avg_f_w_grad_result = avg_fmj_grads_result[0]
        avg_fmj_beta_grads_result = avg_fmj_grads_result[1]
        
        # Update w
        w_new = []
        for i in range(2):
            w_new.append(w_curr[i] - eta * avg_f_w_grad_result[i])
        
        # Update beta
        beta_new = []
        for m in range(M):
            beta_new_m = []
            for i in range(2):
                beta_new_m.append(beta_curr[m][i] - eta * avg_fmj_beta_grads_result[m][i])
            beta_new.append(beta_new_m)
            
        # Update parameters
        w_curr = w_new
        beta_curr = beta_new
        
        # Compute the loss
        loss = loss_fn(w_curr, beta_curr, train_loader_list, 
                       lambda_global, lambda_penal)
        loss_SGD.append(loss)
        
        if epoch == 1 or epoch % repo_step == 0:
            now_time = datetime.datetime.now()
            pass_time = (now_time - start_time).seconds
            print("epoch: {}, loss: {:.10f}, time pass: {}s | SGD {} {} Non-Reparam".format(epoch, loss, pass_time, obj, data_name))
            
    return loss_SGD, w_curr, beta_curr

def train_CD(w0, beta0, n_epochs, devices_train_list, train_loader_list, \
             lambda_global, lambda_penal, repo_step, eta, prob1, obj, data_name, rd_seed=111):
    np.random.seed(rd_seed)
    
    loss_CD = []
    
    w_curr = w0
    beta_curr = beta0

    M = len(beta0)
    
    start_time = datetime.datetime.now()
    for epoch in range(1, n_epochs + 1):
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Compute gradients
        avg_fmj_grads_result = fmj_grads(w_curr, beta_curr, index_choice, 
                                         devices_train_list, lambda_global, lambda_penal)
        avg_fmj_w_grad_result = avg_fmj_grads_result[0]
        avg_fmj_beta_grads_result = avg_fmj_grads_result[1]
        
        # Decide which coordinate to update
        xi = np.random.choice(a=(1,2), p=(prob1, 1-prob1))
        if xi == 1:
            # Update w
            w_new = []
            for i in range(2):
                w_new.append(w_curr[i] - eta * avg_fmj_w_grad_result[i] / prob1)
            w_curr = w_new
        else:
            # Update beta
            beta_new = []
            for m in range(M):
                beta_new_m = []
                for i in range(2):
                    beta_new_m.append(beta_curr[m][i] - eta * avg_fmj_beta_grads_result[m][i] / (1-prob1))
                beta_new.append(beta_new_m)
            beta_curr = beta_new
            
        # Compute the loss
        loss = loss_fn(w_curr, beta_curr, train_loader_list, 
                       lambda_global, lambda_penal)
        loss_CD.append(loss)
        
        if epoch == 1 or epoch % repo_step == 0:
            now_time = datetime.datetime.now()
            pass_time = (now_time - start_time).seconds
            print("epoch: {}, loss: {:.10f}, time pass: {}s | CD {} {} Non-Reparam".format(epoch, loss, pass_time, obj, data_name))
            
    return loss_CD, w_curr, beta_curr

def train_CDVR(w0, beta0, n_communs, devices_train_list, train_loader_list, \
               lambda_global, lambda_penal, repo_step, eta, prob1, rho, obj, data_name, rd_seed=111):
    np.random.seed(rd_seed)
    
    loss_CDVR = []
    
    reported_commun = set()
    
    w_y_curr = w0
    w_v_curr = w0
    
    beta_y_curr = beta0
    beta_v_curr = beta0

    M = len(beta0)
    
    start_time = datetime.datetime.now()
    num_commun = 0  # number of communications happend
    iter_num = 0
    while num_commun < n_communs:
        iter_num += 1
        # Sample j's
        index_choice = []
        for m in range(M):
            index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Sample xi to decide which coordinate to update
        xi = np.random.choice(a=(1,2), p=(prob1, 1-prob1))
    
        # Sample xii to decide whether to update (w_v, beta_v)
        xii = np.random.choice(a=(1,2), p=(rho, 1-rho))
        
        # Compute gradients
        avg_fmj_grads_result_y = fmj_grads(w_y_curr, beta_y_curr, index_choice, 
                                           devices_train_list, lambda_global, lambda_penal)
        avg_fmj_w_grad_result_y = avg_fmj_grads_result_y[0]
        avg_fmj_beta_grads_result_y = avg_fmj_grads_result_y[1]
    
        avg_fmj_grads_result_v = fmj_grads(w_v_curr, beta_v_curr, index_choice, 
                                           devices_train_list, lambda_global, lambda_penal)
        avg_fmj_w_grad_result_v = avg_fmj_grads_result_v[0]
        avg_fmj_beta_grads_result_v = avg_fmj_grads_result_v[1]
        
        # Update gradients involving _v if epoch=1 or xii=1
        if iter_num == 1 or xii == 1:
            F_grads_result_v = F_grads(w_v_curr, beta_v_curr, train_loader_list, 
                                       lambda_global, lambda_penal)
            F_w_grad_result_v = F_grads_result_v[0]
            fm_beta_grads_result_v = F_grads_result_v[1]
            
        # compute g_w_k
        if xi == 1:
            g_w_k = []
            for i in range(2):
                g_w_k.append((avg_fmj_w_grad_result_y[i] - avg_fmj_w_grad_result_v[i])/prob1 + F_w_grad_result_v[i])
        elif xi==2:
            g_w_k = F_w_grad_result_v
            
        # Update w_y
        w_y_new = []
        for i in range(2):
            w_y_new.append(w_y_curr[i] - eta * g_w_k[i])
            
        # If xii=1, update w_v
        if xii == 1:
            w_v_new = w_y_curr
        else:
            w_v_new = w_v_curr
            
        # Compute g_k_beta
        if xi == 1:
            g_k_beta = fm_beta_grads_result_v
        else:
            g_k_beta = []
            for m in range(M):
                g_k_beta_m = []
                for i in range(2):
                    g_k_beta_m.append((avg_fmj_beta_grads_result_y[m][i]-avg_fmj_beta_grads_result_v[m][i])/(1-prob1) + \
                    fm_beta_grads_result_v[m][i])
                g_k_beta.append(g_k_beta_m)
                
        # Update beta_y
        beta_y_new = []
        for m in range(M):
            beta_y_new_m = []
            for i in range(2):
                beta_y_new_m.append(beta_y_curr[m][i] - eta * g_k_beta[m][i])
            beta_y_new.append(beta_y_new_m)
            
        # If xii=1, update beta_v
        if xii == 1:
            beta_v_new = beta_y_curr
        else:
            beta_v_new = beta_v_curr
            
        # Update parameters
        w_y_curr = w_y_new
        beta_y_curr = beta_y_new
        
        w_v_curr = w_v_new
        beta_v_curr = beta_v_new

        if xi == 1:
            num_commun += 1  # When update w, it needs to communicate

            # Compute the loss
            loss = loss_fn(w_y_curr, beta_y_curr, train_loader_list, lambda_global, lambda_penal)
            loss_CDVR.append(loss)
        
        if num_commun == 1 or num_commun % repo_step == 0:
            if num_commun not in reported_commun:
                reported_commun.add(num_commun)
                now_time = datetime.datetime.now()
                pass_time = (now_time - start_time).seconds
                print("num_commun: {}, loss: {:.10f}, time pass: {}s | CDVR {} {} Non-Reparam".format(num_commun, loss, pass_time, obj, data_name))
            
    return loss_CDVR, w_y_curr, beta_y_curr

def loss_fn_local_sgd(w_list, beta_list, train_loader_list, lambda_global, lambda_penal):
    # Define loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')
    
    M = len(w_list)
    
    loss_train = 0.0
    for m in range(M):
        loss_train_local = 0
        
        # Set up global model
        global_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            global_model.weight[:] = w_list[m][0]
            global_model.bias[:] = w_list[m][1]
        
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up training dataset
        train_loader = train_loader_list[m]
        
        # Begin computing the loss
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            
            # Compute loss of global model
            with torch.no_grad():
                outs_global = global_model(imgs.view(imgs.shape[0], -1))
                loss_global = loss_CE_fn(outs_global, labels)
             
            # Compute loss of local model
            with torch.no_grad():
                outs_local = local_model(imgs.view(imgs.shape[0], -1))
                loss_local = loss_CE_fn(outs_local, labels)
                
            # Compute loss of penalty term
            with torch.no_grad():
                p_local = torch.nn.utils.parameters_to_vector(local_model.parameters())
                p_global = torch.nn.utils.parameters_to_vector(global_model.parameters())
                loss_penalty = penalty_fn(p_local, p_global)
                
            # Add all three terms
            with torch.no_grad():
                loss = lambda_global * loss_global + loss_local + (lambda_penal/2) * loss_penalty
            
            # Update local loss_train
            loss_train_local += loss.item()
        
        # Compute average local loss train
        loss_train_local = loss_train_local / len(train_loader)
        
        # Update loss train
        loss_train += loss_train_local
    
    # Compute average loss train
    loss_train = loss_train / M
    
    return loss_train

def fmj_grads_local_sgd(w_list, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal):
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(w_list)
    
    global_models_list = []
    for m in range(M):
        global_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            global_model.weight[:] = w_list[m][0]
            global_model.bias[:] = w_list[m][1]
        global_models_list.append(global_model)
        
    local_models_list = []
    for m in range(M):
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        local_models_list.append(local_model)
    
    loss = 0.0
    for m in range(M):
        train_device = devices_train_list[m]
        img, label = train_device[index_choice[m]]
        local_model = local_models_list[m]
        global_model = global_models_list[m]
        
        img = img.to(device=device)
        label = torch.Tensor([label]).to(device=device).long()
        
        # Compute loss of global model
        out_global = global_model(img.view(1,-1))
        loss_global = loss_CE_fn(out_global, label)
        
        # Compute loss of local model
        out_local = local_model(img.view(1,-1))
        loss_local = loss_CE_fn(out_local, label)
        
        # Compute loss of penalty term
        p_local = torch.nn.utils.parameters_to_vector(local_model.parameters())
        p_global = torch.nn.utils.parameters_to_vector(global_model.parameters())
        loss_penalty = penalty_fn(p_local, p_global)
        
        # Add all three terms
        loss += lambda_global * loss_global + loss_local + (lambda_penal/2) * loss_penalty
    
    loss = loss / M
    
    loss.backward()
    
    w_grads = []
    for m in range(M):
        global_model = global_models_list[m]
        w_grad = []
        for param in global_model.parameters():
            w_grad.append(param.grad)
        w_grads.append(w_grad)
    
    beta_grads = []
    for m in range(M):
        local_model = local_models_list[m]
        beta_grad = []
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grads, beta_grads]

def train_local_sgd(w_list0, beta_list0, sync_step, n_communs, devices_train_list, train_loader_list, 
                    lambda_global, lambda_penal, repo_step, eta, obj, data_name, rd_seed=111):
    np.random.seed(rd_seed)
    
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')
    
    loss_local_sgd = []

    reported_commun = set()
    
    M = len(w_list0)
    
    w_list_curr = w_list0
    beta_list_curr = beta_list0
    
    start_time = datetime.datetime.now()
    num_commun = 0  # number of communications happend
    iter_num = 0
    while num_commun < n_communs:
        iter_num += 1
        # Synchronization
        if iter_num == 1 or iter_num % sync_step == 0:
            sync_weight = torch.zeros(10, 784).to(device)
            sync_bias = torch.zeros(10).to(device)
            
            for m in range(M):
                sync_weight += w_list_curr[m][0]
                sync_bias += w_list_curr[m][1]
                    
            sync_weight = sync_weight / M
            sync_bias = sync_bias / M
                
            for m in range(M):
                w_list_curr[m][0] = sync_weight
                w_list_curr[m][1] = sync_bias

            num_commun += 1  # Add number of communications when sync happens

            # Compute the loss
            loss = loss_fn_local_sgd(w_list_curr, beta_list_curr, train_loader_list, lambda_global, lambda_penal)
            loss_local_sgd.append(loss)
        
        # Sample j's
        index_choice = []
        for m in range(M):
             index_choice.append(np.random.choice(len(devices_train_list[m])))
        
        # Compute gradients
        w_grads, beta_grads = fmj_grads_local_sgd(w_list_curr, beta_list_curr, index_choice, 
                                                  devices_train_list, lambda_global, lambda_penal)
        
        # Compute new w
        w_list_new = []
        for m in range(M):
            w_new = []
            for i in range(2):
                w_new.append(w_list_curr[m][i] - eta * w_grads[m][i])
            w_list_new.append(w_new)
        
        # Compute new beta
        beta_list_new = []
        for m in range(M):
            beta_new = []
            for i in range(2):
                beta_new.append(beta_list_curr[m][i] - eta * beta_grads[m][i])
            beta_list_new.append(beta_new)
        
        # Update parameters
        w_list_curr = w_list_new
        beta_list_curr = beta_list_new
        
        if num_commun == 1 or num_commun % repo_step == 0:
            if num_commun not in reported_commun:
                now_time = datetime.datetime.now()
                pass_time = (now_time - start_time).seconds
                print("num_commun: {}, loss: {:.10f}, time pass: {}s | Local_SGD {} {} Non-Reparam".format(num_commun, loss, pass_time, obj, data_name))
            
    return loss_local_sgd, w_list_curr, beta_list_curr