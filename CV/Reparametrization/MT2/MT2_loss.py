import numpy as np
import torch
import torch.nn as nn
import datetime
import time

def loss_fn(w, beta_list, train_loader_list, lambda_global, lambda_penal, device):
    # Define loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')
    
    M = len(beta_list)

    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0] / np.sqrt(M)
        global_model.bias[:] = w[1] / np.sqrt(M)
    
    loss_train = 0.0
    for m in range(M):
        loss_train_local = 0.0
        
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up training dataset
        train_loader = train_loader_list[m]
        
        # Begin computing the loss
        n_count = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
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
        loss_train_local = loss_train_local / n_count
        
        # Update loss train
        loss_train += loss_train_local
    
    # Compute average loss train
    loss_train = loss_train / M
    
    return loss_train

def loss_fn_norescale(w, beta_list, train_loader_list, lambda_global, lambda_penal, device):
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
        loss_train_local = 0.0
        
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up training dataset
        train_loader = train_loader_list[m]
        
        # Begin computing the loss
        n_count = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
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
        loss_train_local = loss_train_local / n_count
        
        # Update loss train
        loss_train += loss_train_local
    
    # Compute average loss train
    loss_train = loss_train / M
    
    return loss_train

def test_accu_fn(w, beta_list, val_loader_list, device):
    softmax_model = nn.Softmax(dim=1)

    M = len(beta_list)

    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0] / np.sqrt(M)
        global_model.bias[:] = w[1] / np.sqrt(M)
    
    accu = 0.0
    for m in range(M):
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up validation dataset
        val_loader = val_loader_list[m]
        
        # Begin computing the accuracy
        right_count = 0
        n_count = 0
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
            # Compute the number of right cases
            with torch.no_grad():
                outs = softmax_model(local_model(imgs.view(imgs.shape[0], -1)))
                outs = torch.argmax(outs, dim=1)
                right_count += (outs == labels).sum().item()
        accu += right_count / n_count
    
    return accu / M

def test_accu_fn_norescale(w, beta_list, val_loader_list, device):
    softmax_model = nn.Softmax(dim=1)

    M = len(beta_list)

    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0]
        global_model.bias[:] = w[1]
    
    accu = 0.0
    for m in range(M):
        # Set up the local model
        local_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            local_model.weight[:] = beta_list[m][0]
            local_model.bias[:] = beta_list[m][1]
        
        # Set up validation dataset
        val_loader = val_loader_list[m]
        
        # Begin computing the accuracy
        right_count = 0
        n_count = 0
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
            # Compute the number of right cases
            with torch.no_grad():
                outs = softmax_model(local_model(imgs.view(imgs.shape[0], -1)))
                outs = torch.argmax(outs, dim=1)
                right_count += (outs == labels).sum().item()
        accu += right_count / n_count
    
    return accu / M

def F_grads(w, beta_list, train_loader_list, lambda_global, lambda_penal, device):
    # Define loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(beta_list)
    
    # Set up global model
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0] / np.sqrt(M)
        global_model.bias[:] = w[1] / np.sqrt(M)
    
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
        n_count = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
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
        loss_device = loss_device / n_count
        
        # Update overall loss
        loss += loss_device
    
    # Average overall loss
    loss = loss / M
    
    # Compute gradients
    loss.backward()
    
    # Store the gradients w.r.t w
    w_grad = []
    for param in global_model.parameters():
        w_grad.append(param.grad / np.sqrt(M))
        
    # Store the gradients w.r.t. beta_m's
    beta_grads = []
    for m in range(M):
        local_model = local_models_list[m]
        beta_grad = []
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grad, beta_grads]

def F_grads_norescale(w, beta_list, train_loader_list, lambda_global, lambda_penal, device):
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
        n_count = 0
        # batch_index = np.random.choice(len(train_loader)-1)
        # k_batch = 0
        for imgs, labels in train_loader:
            # k_batch += 1
            # if k_batch != (batch_index+1):
            #     continue
            
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_count += len(labels)
            
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
        loss_device = loss_device / n_count
        #loss_device = loss_device / len(train_loader)
        
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

def fmj_grads(w, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal, device):
    # Return: A list of grads w.r.t. beta_m's
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(beta_list)
    
    global_model = nn.Linear(784, 10).to(device=device)
    with torch.no_grad():
        global_model.weight[:] = w[0] / np.sqrt(M)
        global_model.bias[:] = w[1] / np.sqrt(M)
        
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
        w_grad.append(param.grad / np.sqrt(M))
    
    beta_grads = []
    for m in range(M):
        beta_grad = []
        local_model = local_models_list[m]
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grad, beta_grads]

def fmj_grads_norescale(w, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal, device):
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

def fmj_grads_LSGD(w_list, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal, device):
    loss_CE_fn = nn.CrossEntropyLoss()
    penalty_fn = nn.MSELoss(reduction='sum')

    M = len(w_list)
    
    global_models_list = []
    for m in range(M):
        global_model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            global_model.weight[:] = w_list[m][0] / np.sqrt(M)
            global_model.bias[:] = w_list[m][1] / np.sqrt(M)
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
            w_grad.append(param.grad / np.sqrt(M))
        w_grads.append(w_grad)
    
    beta_grads = []
    for m in range(M):
        local_model = local_models_list[m]
        beta_grad = []
        for param in local_model.parameters():
            beta_grad.append(param.grad)
        beta_grads.append(beta_grad)
    
    return [w_grads, beta_grads]

def fmj_grads_LSGD_norescale(w_list, beta_list, index_choice, devices_train_list, lambda_global, lambda_penal, device):
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