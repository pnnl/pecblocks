import torch
import pandas as pd
import numpy as np
import os, sys
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoChannelWiseNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    lr = 1e-3 # original: 1e-3
    num_iter = 1000 # original: 40000
    test_freq = 100
    # n_batch = 1 # TODO: figure out what this indicates
    n_b = 4 # original: 2
    n_a = 4 # original: 2
    _n_hidden = 10

    model_folder = './pkl_models'
    model_struct = 'FGF' # could be FG, GF, FGF, GFG

    # Keywords of column names in the dataset
    COL_T = ['TIME']
    COL_X = [ 'Id_pu_sys','Iq_pu_sys']
    COL_U = ['Udc', 'Idc','Vd_pu_sys','Vq_pu_sys','f_Inv1']

    df_training = pd.read_hdf ('new.hdf5')

    print(df_training.describe())
    quit()

    time_arr = np.array(df1[COL_T], dtype=np.float32)
    t_step = np.mean(np.diff(time_arr.ravel())) #time_exp[1] - time_exp[0]
    
    y = np.array(df1[COL_X], dtype=np.float32)
    u = np.array(df1[COL_U], dtype=np.float32)
    
    N = df1[COL_T].size
    t = np.arange(N)*t_step

    # scale state
    u = u / np.array([1.0, 1.0,1.0,1.0,60.0]) # scale the frequency to pu.

    # Prepare data
    u_torch = torch.tensor(u[None, :, :], dtype=torch.float, requires_grad=False)
    y_true_torch = torch.tensor(y[None, :, :], dtype=torch.float)

    in_size = len(COL_U)
    out_size = len(COL_X)

    # 'FGF'
    nn_static1 = MimoChannelWiseNonLinearity(channels=in_size, n_hidden=_n_hidden)
    G = MimoLinearDynamicalOperator(in_channels=in_size, out_channels=out_size, n_b=n_b, n_a=n_a, n_k=1)
    nn_static2 = MimoChannelWiseNonLinearity(channels=out_size, n_hidden=_n_hidden)
    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': nn_static1.parameters(), 'lr': lr},
        {'params': G.parameters(), 'lr': lr},
        {'params': nn_static2.parameters(), 'lr': lr},
    ], lr=lr)

    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):
        optimizer.zero_grad()
        y_nl = nn_static1(u_torch)
        y_lin = G(y_nl)
        y_hat = nn_static2(y_lin)
        # Compute fit loss
        # err_fit = y_meas_torch - y_hat
        err_fit = y_true_torch - y_hat
        #loss_fit = torch.mean(err_fit**2)
        loss_fit = torch.sum(torch.abs(err_fit))
        loss = loss_fit
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')
        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    torch.save(nn_static1.state_dict(), os.path.join(model_folder, "F1.pkl"))
    torch.save(G.state_dict(), os.path.join(model_folder, "G.pkl"))
    torch.save(nn_static2.state_dict(), os.path.join(model_folder, "F2.pkl"))

    y_hat = y_hat.detach().numpy()[0, :, :]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y[:, 0], 'k', label="$y$")
    ax[0].plot(t, y_hat[:, 0], 'b', label="$\hat y$")
    ax[0].legend()
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Id(pu)')
    ax[0].grid()

    ax[1].plot(t, y[:, 1], 'k', label="$y$")
    ax[1].plot(t, y_hat[:, 1], 'b', label="$\hat y$")
    ax[1].legend()
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Iq(pu)')
    ax[1].grid()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)
    plt.savefig(os.path.join(model_folder,'Inverter_train_loss.pdf'))
    plt.show()