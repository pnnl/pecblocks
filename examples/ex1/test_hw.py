# Copyright (C) 2021 Battelle Memorial Institute

from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from torch.nn.modules import activation

import torch
from dynonet.lti import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import dynonet.metrics
import pecblocks.util

from scipy import signal

training_sets = [
   {'Title':'Irrad to Vdc',
    'model_name':'GtoVdc',
    'Path':'./data/training1.zip',
    'tmin':3.5,
    'tmax':5.0,
    'n_a': 8,
    'n_b': 8,
    'n_k': 1,
    'n_skip': 1,
    'nh_1': 20,
    'nh_2': 20,
    'decimate': 1000,
    'u':'MODELS IRROUT_I-branch',
    'y':'VCAP_V-branch'},
   {'Title':'Irrad to Idc',
    'model_name':'GtoIdc',
    'Path':'./data/training1.zip',
    'tmin':3.5,
    'tmax':5.0,
    'n_a': 1,
    'n_b': 1,
    'n_k': 1,
    'n_skip': 1,
    'nh_1': 20,
    'nh_2': 20,
    'decimate': 1000,
    'u':'MODELS IRROUT_I-branch',
    'y':'DCND_I-branch'},
   {'Title':'Temp to Vdc',
    'model_name':'TtoVdc',
    'Path':'./data/training2.zip',
    'tmin':3.5,
    'tmax':5.0,
    'n_a': 2,
    'n_b': 2,
    'n_k': 1,
    'n_skip': 1,
    'nh_1': 20,
    'nh_2': 20,
    'decimate': 1000,
    'u':'MODELS TMPOUT_I-branch',
    'y':'VCAP_V-branch'},
   {'Title':'Temp to Idc',
    'model_name':'TtoIdc',
    'Path':'./data/training2.zip',
    'tmin':3.5,
    'tmax':5.0,
    'n_a': 1,
    'n_b': 1,
    'n_k': 0,
    'n_skip': 1,
    'nh_1': 2,
    'nh_2': 2,
    'decimate': 1000,
    'u':'MODELS TMPOUT_I-branch',
    'y':'DCND_I-branch'},
   {'Title':'Vdc to Idc',
    'model_name':'VdctoIdc',
    'Path':'./data/average_irradiance.zip',
    'tmin':2.5,
    'tmax':4.0,
    'n_a': 8,
    'n_b': 8,
    'n_k': 0,
    'n_skip': 1,
    'nh_1': 20,
    'nh_2': 20,
    'decimate': 1000,
    'u':'VDC_V-node',
    'y':'XX0004 DCND_I-branch'},
    {'Title':'Vdc to Pac',
     'model_name':'VdctoPac',
     'Path':'./data/average_irradiance.zip',
     'tmin':2.7,
     'tmax':3.3,
     'n_a': 8,
     'n_b': 8,
     'n_k': 0,
     'n_skip': 2,
     'nh_1': 20,
     'nh_2': 20,
     'decimate': 1000,
     'u':'VDC_V-node',
     'y':'MODELS PAC_I-branch'},
    {'Title':'Vdc to Qac',
     'model_name':'VdctoQac',
     'Path':'./data/average_irradiance.zip',
     'tmin':2.7,
     'tmax':3.3,
     'n_a': 8,
     'n_b': 8,
     'n_k': 0,
     'n_skip': 2,
     'nh_1': 20,
     'nh_2': 20,
     'decimate': 1000,
     'u':'VDC_V-node',
     'y':'MODELS QAC_I-branch'}
]

def process_training_set (row):
    data_path = row['Path']

    seed = 11
    np.random.seed(seed)
    torch.manual_seed(seed)

    # In[Settings]
    lr_ADAM = 5e-4
    num_iter = 3500
    msg_freq = 100
    decimate = row['decimate']
    n_skip = row['n_skip']
    n_b = row['n_b']
    n_a = row['n_a']
    n_k = row['n_k']
    nh_1 = row['nh_1']
    nh_2 = row['nh_2']
    model_name = row['model_name']
    model_type = 'FGF'
    t_step = 0.00002

    df_training = pecblocks.util.read_csv_files(data_path, pattern=".csv")
    df1 = df_training[(df_training['time']>row['tmin']) & (df_training['time']<row['tmax'])]

#    df_u = df1.filter(like='MODELS TMPOUT_I-branch', axis=1)
    df_u = df1.filter(like=row['u'], axis=1)
    df_y = df1.filter(like=row['y'], axis=1)
    df_y = df_y.sum(axis=1)
    df_u = df_u.sum(axis=1)
    y = np.transpose(df_y.to_numpy(dtype=np.float32))/1000.0  # batch, time, channel
    u = np.transpose(df_u.to_numpy(dtype=np.float32))/1000.0 # 1.0  # normalization of T or G
    t = np.arange(0,  (y.size)*t_step, t_step, dtype=np.float32)
    y = y.reshape(y.size,1)
    u = u.reshape(u.size,1)
    t = t.reshape(t.size,1)

    n_fit = y.size-1
    t0 = t[0][0]
    u0 = u[0][0]
    y0 = y[0][0]
    print ('n={:d},dec={:d},skip={:d},t0={:.4f},u0={:.4f},y0={:.4f}'.format (n_fit, decimate, n_skip, t0, u0, y0))
    y = y - y0
    u = u - u0
    print ('umin/max={:.4f} {:.4f}, ymin/max={:.4f} {:.4f}'.format (np.min(u), np.max(u), np.min(y), np.max(y)))
    scale_u = 1.0 / (np.max(u) - np.min(u))
    scale_y = 1.0 / (np.max(y) - np.min(y))
    y = y * scale_y
    u = u * scale_u

    # In[Fit data, slicing arrays to customize the sampling rate]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    u_fit_torch = torch.tensor(u_fit[None, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :], dtype=torch.float)

    # In[Prepare model]
    F1 = SisoStaticNonLinearity(n_hidden=nh_1, activation='tanh') # Either 'tanh', 'relu', or 'sigmoid'. Default: 'tanh'
    G1 = SisoLinearDynamicalOperator(n_b, n_a, n_k)
    F2 = SisoStaticNonLinearity(n_hidden=nh_2, activation='tanh')

    G1_num, G1_den = G1.get_tfdata()

    print("Initial G1", G1_num, G1_den)

    def model(u_in):
        y1_nl = F1(u_fit_torch)
        y2_lin = G1(y1_nl)
        y_hat = F2(y2_lin)
        return y_hat, y2_lin, y1_nl

    # In[Setup optimizer]
    params = [
        {'params': G1.parameters(), 'lr': lr_ADAM},
        {'params': F2.parameters(), 'lr': lr_ADAM},
        {'params': F1.parameters(), 'lr': lr_ADAM}
    ]
    optimizer = torch.optim.Adam(params, lr=lr_ADAM) # , amsgrad=True)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Simulate
        y_hat, y2_lin, y1_nl = model(u_fit_torch)

        err_fit = y_fit_torch[:, n_skip:, :] - y_hat[:, n_skip:, :]
        #loss = torch.mean(err_fit**2)*100  # orginal
        #loss = torch.mean(err_fit**2) + 10*torch.max(err_fit**2)
        loss = torch.sum(torch.abs(err_fit))
        # loss = torch.sum(err_fit**2)

        # Backward pas
        loss.backward()

        LOSS.append(loss.item())
        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Iter {itr} | Fit Loss {loss:.6f} | RMSE:{RMSE:.4f}')
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Save model]
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
    torch.save(F2.state_dict(), os.path.join(model_folder, "F2.pkl"))

    G1_num, G1_den = G1.get_tfdata()

    print("Trained G1", G1_num, G1_den)
    Hz = signal.TransferFunction (G1_num, G1_den, dt=decimate * t_step)
    max_pole = np.amax (np.absolute(Hz.poles))
    print ('Transfer Function has step', Hz.dt, 'largest pole', max_pole)
    if max_pole > 1.0:
        print ('************* G1 is not stable ************')
#    print (Hz.poles)
#    print (Hz.zeros)

    # In[Simulate one more time]
    with torch.no_grad():
        y_hat, y2_lin, y1_nl = model(u_fit_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y2_lin = y2_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    e_rms = dynonet.metrics.error_rmse(y_fit[n_skip:], y_hat[n_skip:])[0]
    fit_idx = dynonet.metrics.fit_index(y_fit[n_skip:], y_hat[n_skip:])[0]
    r_sq = dynonet.metrics.r_squared(y_fit[n_skip:], y_hat[n_skip:])[0]
    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")

    # In[Plot]
    plt.figure()
    plt.title('{:s}; FGF, Fit={:.2f}%, h={:d}, n={:d}, k={:d}, dec={:d}'.format(row['Title'],
                                                                                fit_idx, 
                                                                                max(nh_1,nh_2),
                                                                                max(n_a, n_b), 
                                                                                n_k, 
                                                                                decimate))
    plt.plot(t_fit[n_skip:], u_fit[n_skip:], 'r', label="$u$")
    plt.plot(t_fit[n_skip:], y_fit[n_skip:], 'k', label="$y$")
    plt.plot(t_fit[n_skip:], y_hat[n_skip:], 'b', label="$\hat y$")
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized [pu]')
    plt.legend()

    plt.savefig(os.path.join(model_folder,'{:s}_train_fit.pdf'.format(model_name)))
    plt.show()

#    plt.figure()

#    plt.plot(t_fit[n_skip:], u_fit[n_skip:], 'g', label="$u$")
#    plt.legend()

#    plt.savefig(os.path.join(model_folder,'test_hw_train_input_u.pdf'))
#    plt.show()

    # In[Plot loss]
#    plt.figure()
#    plt.plot(LOSS)
#    plt.grid(True)
    
#    plt.savefig(os.path.join(model_folder,'test_hw_train_loss.pdf'))
#    plt.show()

if __name__ == '__main__':
    for row in training_sets:
        process_training_set (row)
#        quit()

