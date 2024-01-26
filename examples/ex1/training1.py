# Copyright (C) 2021 Battelle Memorial Institute
'''
 #TODO
 #  1. save the best model during training
 # 
'''
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from torch.nn.modules import activation
import dynonet
import pecblocks.util

import torch
from dynonet.lti import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import dynonet.metrics

data_path = r'./data/training1.zip'

if __name__ == '__main__':
# In[Column names in the dataset]
 # In[Set seed for reproducibility]
    seed = 11
    np.random.seed(seed)
    torch.manual_seed(seed)

    # In[Settings]
    lr_ADAM = 5e-4
    lr_BFGS = 1e0
    num_iter_ADAM = 2000  # ADAM iterations 20000
    num_iter_BFGS = 0  # final BFGS iterations
    msg_freq = 100
    n_fit = 54000
    decimate = 1
    n_skip = 0
    n_batch = 1
    n_b = 2 # numerator
    n_a = 3 # denominator
    model_name = 'test'

    num_iter = num_iter_ADAM + num_iter_BFGS

    model_type = "FG" # or "GFG"

    t_step = 0.00002
    # t_start = 2.9
    # t_end = 3.7

    COL_T = ['time']
    COL_U = ['temp']
    COL_Y = ['Pac']

    df_training = pecblocks.util.read_csv_files(data_path, pattern=".csv")

#    print (df_training.describe())
    print ('df_training columns', df_training.columns)
    print ('df_training shape', df_training.shape)

    df1 = df_training[(df_training['time']>3.5) & (df_training['time']<5.0)]

    df_u = df1.filter(like='MODELS IRROUT_I-branch', axis=1)
    df_y = df1.filter(like='VCAP_V-branch', axis=1)

    # print(df_y.describe())
    # print(df_u.describe())

    df_y = df_y.sum(axis=1)
    df_u = df_u.sum(axis=1)

    # In[Load dataset]

    # Extract data
    y = np.transpose(df_y.to_numpy(dtype=np.float32))/1000.0  # batch, time, channel
    u = np.transpose(df_u.to_numpy(dtype=np.float32))/1000.0 # 1.0  # normalization of T or G
    t = np.arange(0,  (y.size)*t_step, t_step, dtype=np.float32)

    y = y.reshape(y.size,1)
    u = u.reshape(u.size,1)
    t = t.reshape(t.size,1)

    n_fit = y.size-1

    # In[Fit data, slicing arrays to customize the sampling rate]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    u_fit_torch = torch.tensor(u_fit[None, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :], dtype=torch.float)

    # In[Prepare model]
    G1 = SisoLinearDynamicalOperator(n_b, n_a, n_k=1)
    F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh') # Either 'tanh', 'relu', or 'sigmoid'. Default: 'tanh'
    #F_nl = SisoStaticPwNonLinearity(n_hidden=10, num_break_points=6)
    G2 = SisoLinearDynamicalOperator(n_b, n_a,n_k=0)

    G2_num, G2_den = G2.get_tfdata()

    print("Initial G2_num, G2_den", G2_num, G2_den)

    def model(u_in):
        if model_type == "FG":
            y1_nl = F_nl(u_fit_torch)
            y_hat = G2(y1_nl)
            return y_hat, y1_nl
        elif model_type == "GFG":
            y1_lin = G1(u_fit_torch)
            y1_nl = F_nl(y1_lin)
            #y1_nl = F_nl(u_fit_torch)
            y_hat = G2(y1_nl)
            return y_hat, y1_nl , y1_lin
        else:
           return None

    # In[Setup optimizer]
    params = [
        {'params': G1.parameters(), 'lr': lr_ADAM},
        {'params': G2.parameters(), 'lr': lr_ADAM},
        {'params': F_nl.parameters(), 'lr': lr_ADAM},
    ]

    if model_type == "FG":
        params = [
        # {'params': G1.parameters(), 'lr': lr_ADAM},
        {'params': G2.parameters(), 'lr': lr_ADAM},
        {'params': F_nl.parameters(), 'lr': lr_ADAM},
    ]
    

    optimizer_ADAM = torch.optim.Adam(params, lr=lr_ADAM, amsgrad=True)
    

    #optimizer_LBFGS = torch.optim.LBFGS(list(G1.parameters()) + list(G2.parameters()) + list(F_nl.parameters()), lr=lr_BFGS)
    optimizer_LBFGS = torch.optim.LBFGS( list(G2.parameters()) + list(F_nl.parameters()), lr=lr_BFGS)

    def closure():
        optimizer_LBFGS.zero_grad()

        # Simulate
        if model_type == "FG":
            y_hat, y1_nl = model(u_fit_torch)
        elif model_type == "GFG":
            y_hat, y1_nl, y1_lin = model(u_fit_torch)
        

        # Compute fit loss
        err_fit = y_fit_torch[:, n_skip:, :] - y_hat[:, n_skip:, :]
        #loss = torch.mean(err_fit**2)*100  # orginal
        #loss = torch.mean(err_fit**2) + 10*torch.max(err_fit**2)
        loss = torch.sum(torch.abs(err_fit))
        # loss = torch.sum(err_fit**2)

        # Backward pas
        loss.backward()
        return loss


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        if itr < num_iter_ADAM:
            msg_freq = 10
            loss_train = optimizer_ADAM.step(closure)
        else:
            msg_freq = 10
            loss_train = optimizer_LBFGS.step(closure)

        LOSS.append(loss_train.item())
        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss_train)
            print(f'Iter {itr} | Fit Loss {loss_train:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Save model]
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if model_type == "GFG":
        torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join(model_folder, "F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    G2_num, G2_den = G2.get_tfdata()

    print("Trained G2_num, G2_den", G2_num, G2_den)
    print("Trained F_nl state_dict", F_nl.state_dict())

    # In[Simulate one more time]
    with torch.no_grad():
        if model_type == "FG":
            y_hat, y1_nl = model(u_fit_torch)
        elif model_type == "GFG":
            y_hat, y1_nl, y1_lin = model(u_fit_torch)
        

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    #y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    plt.plot(t_fit[n_skip:], y_fit[n_skip:], 'k', label="$y$")
    plt.plot(t_fit[n_skip:], y_hat[n_skip:], 'b', label="$\hat y$")
    plt.legend()

    plt.savefig(os.path.join(model_folder,'training1_fit.pdf'))
    plt.show()

    plt.figure()

    plt.plot(t_fit[n_skip:], u_fit[n_skip:], 'g', label="$u$")
    plt.legend()

    plt.savefig(os.path.join(model_folder,'training1_input_u.pdf'))
#    plt.show()

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)
    
    plt.savefig(os.path.join(model_folder,'training1_loss.pdf'))
#    plt.show()

    


