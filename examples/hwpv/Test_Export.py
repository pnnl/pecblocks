import torch
import pandas as pd
import numpy as np
import os, sys
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoChannelWiseNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import control
import pecblocks.util
import json

testing_data_dir = './data/inverter_loadChange'

def slice_tensor(B, key, idx=0):
    t = B[key]
    shape = list(t.shape)
    ndim = len(shape)
    if ndim == 1:
        return t[:].numpy()
    elif ndim == 2:
        if idx == 0:
            return t[:,0].numpy()
        elif idx == 1:
            return t[0,:].numpy()
        else:
            print (key, shape, 'unsupported index', idx)
    elif ndim == 3:
        if idx == 0:
            return t[:,0,0].numpy()
        elif idx == 1:
            return t[0,:,0].numpy()
        elif idx == 2:
            return t[0,0,:].numpy()
        else:
            print (key, shape, 'unsupported index', idx)
    else:
        print (key, shape, 'too many dimensions')
    return None

def append_net(model, nchan, block):
    for i in range(nchan):
        key = 'net.{:d}.0.weight'.format(i)
        ary = slice_tensor (block, key, 0)
        model[key] = ary.tolist()

        key = 'net.{:d}.0.bias'.format(i)
        ary = slice_tensor (block, key, 0)
        model[key] = ary.tolist()

        key = 'net.{:d}.2.weight'.format(i)
        ary = slice_tensor (block, key, 1)
        model[key] = ary.tolist()

        key = 'net.{:d}.2.bias'.format(i)
        ary = slice_tensor (block, key, 0)
        model[key] = ary.tolist()

def append_lti(model, G, block):
    n_in = G.in_channels
    n_out = G.out_channels
    a = block['a_coeff']
    b = block['b_coeff']
#    print ('a_coeff shape:', a.shape) # should be (n_out, n_in, n_a==n_b)
    for i in range(n_in):
        for j in range(n_out):
            key = 'a_{:d}_{:d}'.format(i, j)
            ary = a[j,i,:].numpy()
            model[key] = ary.tolist()

            key = 'b_{:d}_{:d}'.format(i, j)
            ary = b[j,i,:].numpy()
            model[key] = ary.tolist()

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings

    test_freq = 100
    n_batch = 1 # TODO: figure out what this indicates
    n_b = 4 # original: 2
    n_a = 4 # original: 2
    _n_hidden = 10

    t_step = 0.0005

    model_name = "inverter_FGF"

    model_folder = os.path.join(os.path.dirname(__file__),"models/"+ model_name)

    model_struct = 'FGF' # could be FG, GF, FGF, GFG

    # Keywords of column names in the dataset
    COL_T = ['TIME']
    COL_X = ['Id_pu_sys','Iq_pu_sys']
    COL_U = ['Udc', 'Idc','Vd_pu_sys','Vq_pu_sys','f_Inv1']

    df_training = pecblocks.util.read_csv_files(testing_data_dir, pattern="ds5.csv")

    # skip the first 1 second as it is used for initialization
    df1 = df_training[(df_training['TIME']>1.5) & (df_training['TIME']<4.5)]

    t = np.array(df1[COL_T], dtype=np.float32)
    y_meas = np.array(df1[COL_X], dtype=np.float32)
    u = np.array(df1[COL_U], dtype=np.float32)

    # scale state
    u = u / np.array([1.0, 1.0,1.0,1.0,60.0]) # scale the frequency to pu.

    # Prepare data
    u_torch = torch.tensor(u[None, :, :], dtype=torch.float, requires_grad=False)
    y_true_torch = torch.tensor(y_meas[None, :, :], dtype=torch.float)

    in_size = len(COL_U)
    out_size = len(COL_X)

    B1 = torch.load(os.path.join(model_folder, "G.pkl"))
    B2 = torch.load(os.path.join(model_folder, "F1.pkl"))
    B3 = torch.load(os.path.join(model_folder, "F2.pkl"))

    # model structure 'FGF'
    nn_static1 = MimoChannelWiseNonLinearity(channels=in_size, n_hidden=_n_hidden)
    G = MimoLinearDynamicalOperator(in_channels=in_size, out_channels=out_size, n_b=n_b, n_a=n_a, n_k=1)
    nn_static2 = MimoChannelWiseNonLinearity(channels=out_size, n_hidden=_n_hidden)
    G.load_state_dict(B1)
    nn_static1.load_state_dict(B2)
    nn_static2.load_state_dict(B3)

    models = {'name':'Inv', 'type':'F1+G1+F2', 't_step': t_step}
    models ['G1'] = {'n_in': G.in_channels, 'n_out': G.out_channels, 'n_b': G.n_b, 'n_a': G.n_a, 'n_k': G.n_k}
    append_lti (models['G1'], G, B1)
    models ['F1'] = {'n_chan': G.in_channels, 'n_h': _n_hidden, 'activation': 'relu'}
    append_net (models['F1'], G.in_channels, B2)
    models ['F2'] = {'n_chan': G.out_channels, 'n_h': _n_hidden, 'activation': 'relu'}
    append_net (models['F2'], G.out_channels, B3)

    with open ('models.json', 'w') as write_file:
        json.dump (models, write_file, indent=4, sort_keys=True)

    with torch.no_grad():
        #FGF model
        y_nl = nn_static1(u_torch)
        y_lin = G(y_nl)
        y_hat =nn_static2(y_lin)

    y_nl = y_nl.detach().numpy()[0, :, :]
    y_lin = y_lin.detach().numpy()[0, :, :]
    y_hat = y_hat.detach().numpy()[0, :, :]

#    quit()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y_meas[:, 0], 'k', label="$y$")
    ax[0].plot(t, y_hat[:, 0], 'r', label="$\hat y$")
    ax[0].plot(t, y_meas[:, 0] - y_hat[:, 0], 'g', label="$e$")
    ax[0].set_xlim(1.8, 4.5)
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Id(pu)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y_meas[:, 1], 'k', label="$y$")
    ax[1].plot(t, y_hat[:, 1], 'r', label="$\hat y$")
    ax[1].plot(t, y_meas[:, 1] - y_hat[:, 1], 'g', label="$e$")
    ax[1].legend()
    ax[1].set_xlim(1.8, 4.5)
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Iq(pu)')
    ax[1].grid()
    plt.show()

    # In[Inspect linear model]

    # First linear block
    a_coeff_1 = G.a_coeff.detach().numpy()
    b_coeff_1 = G.b_coeff.detach().numpy()
    a_poly_1 = np.empty_like(a_coeff_1, shape=(out_size, in_size, n_a + 1))
    a_poly_1[:, :, 0] = 1
    a_poly_1[:, :, 1:] = a_coeff_1[:, :, :]
    b_poly_1 = np.array(b_coeff_1)
    G1_sys = control.TransferFunction(b_poly_1, a_poly_1, t_step)

    plt.figure()
    mag_G1_1, phase_G1_1, omega_G1_1 = control.bode(G1_sys[0, 0])
    plt.show()
