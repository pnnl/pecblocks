import pandas as pd
import numpy as np
import os
import sys
import time
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import dynonet.metrics
from common import PVInvDataset
import pecblocks.util
import json
import torch
import torch.nn as nn
import math

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

idx_in = [0,1,2,3,4,5,6]
idx_out = [7,8,9]

class pv1():
  def __init__(self, filename):
    fp = open (filename, 'r')
    config = json.load (fp)
    fp.close()
    self.lr = config['lr']
    self.num_iter = config['num_iter']
    self.print_freq = config['print_freq']
    self.batch_size = config['batch_size']
    self.n_skip = config['n_skip']
    self.n_trunc = config['n_trunc']
    self.n_dec = config['n_dec']
    self.na = config['na']
    self.nb = config['nb']
    self.nk = config['nk']
    self.activation = config['activation']
    self.nh1 = config['nh1']
    self.nh2 = config['nh2']
    self.COL_T = config['COL_T']
    self.COL_Y = config['COL_Y']
    self.COL_U = config['COL_U']
    self.ctl_mode_t = config['mode_t']
    self.ctl_mode_y = config['mode_y']

    self.data_train = None
    self.normfacs = None
    self.t = None
    self.n_cases = 0
    self.t_step = 1.0e-3

  def append_net(self, model, label, F):
    block = F.state_dict()
    n_in = F.net[0].in_features
    n_hid = F.net[0].out_features
    n_out = F.net[2].out_features
    activation = str(F.net[1]).lower().replace("(", "").replace(")", "")
    model[label] = {'n_in': n_in, 'n_hid': n_hid, 'n_out': n_out, 'activation': activation}
    key = 'net.0.weight'
    model[label][key] = block[key][:,:].numpy().tolist()
    key = 'net.0.bias'
    model[label][key] = block[key][:].numpy().tolist()
    key = 'net.2.weight'
    model[label][key] = block[key][:,:].numpy().tolist()
    key = 'net.2.bias'
    model[label][key] = block[key][:].numpy().tolist()

  def append_lti(self, model, label, H):
    n_in = H.in_channels
    n_out = H.out_channels
    model[label] = {'n_in': n_in, 'n_out': n_out, 'n_b': H.n_b, 'n_a': H.n_a, 'n_k': H.n_k}
    block = H.state_dict()
    a = block['a_coeff']
    b = block['b_coeff']
  #  print ('a_coeff shape:', a.shape) # should be (n_out, n_in, n_a==n_b)
    for i in range(n_in):
      for j in range(n_out):
        key = 'a_{:d}_{:d}'.format(i, j)
        ary = a[j,i,:].numpy()
        model[label][key] = ary.tolist()

        key = 'b_{:d}_{:d}'.format(i, j)
        ary = b[j,i,:].numpy()
        model[label][key] = ary.tolist()

  def loadTrainingData(self, data_path):
    df_list = pecblocks.util.read_hdf5_file (data_path, self.COL_T + self.COL_Y + self.COL_U, 
                                             self.n_dec, self.n_skip, self.n_trunc)
    print ('read', len(df_list), 'dataframes')

    # get the len of data of interest 
    df_1 = df_list[0]
    time_arr = np.array(df_1[self.COL_T], dtype=np.float32)
    self.t_step = np.mean(np.diff(time_arr.ravel())) #time_exp[1] - time_exp[0]
    data_len = df_1[self.COL_T].size
    self.t = np.arange(data_len)*self.t_step

    n_input_output = len(self.COL_U) + len(self.COL_Y)

    # construct a control mode signal
    mode_sig = np.zeros (data_len)
    for i in range(data_len):
      mode_sig[i] = np.interp (self.t[i], self.ctl_mode_t, self.ctl_mode_y)

    # organize the training dataset by scenario, time and features
    self.n_cases = len(df_list)
    data_mat = np.empty((self.n_cases,data_len,n_input_output+2)) # extra space for derived features
    print ('dt={:.6f} data_len={:d} n_io={:d} n_case={:d}'.format (self.t_step, data_len, n_input_output, self.n_cases))
    for sc_idx in range(self.n_cases):
      df_data = df_list[sc_idx][self.COL_U+self.COL_Y]
      df_poly = 0.001 * df_data['G'] * df_data['Vrms']
      df_mode = pd.DataFrame (data=mode_sig)
      df_data.insert(len(self.COL_U), 'GVrms', df_poly)
      df_data.insert(len(self.COL_U)+1, 'Mode', df_mode)
      data_mat[sc_idx,:,:] = np.array(df_data)
    self.data_train = data_mat.astype(np.float32)
    self.COL_U.append('GVrms')
    self.COL_U.append('Mode')
    print (self.COL_U, self.COL_Y, self.data_train.shape)

  def applyAndSaveNormalization(self, model_folder):
    # Normalize the data; save the normalization factors
    self.normfacs = {}
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)
    print ('shapes of t', self.t.shape, 'data_train', self.data_train.shape, 
           'n_in={:d}, n_out={:d}'.format (in_size, out_size))
    print ('t range {:.6f} to {:.6f}'.format (self.t[0], self.t[-1]))
    print ('Before Scaling:')
    print ('Column       Min       Max      Mean     Range')
    idx = 0
    for c in self.COL_U + self.COL_Y:
      dmax = np.max (self.data_train[:,:,idx])
      dmin = np.min (self.data_train[:,:,idx])
      dmean = np.mean (self.data_train[:,:,idx]) # mean over scenarios and time
      drange = dmax - dmin
      print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, dmin, dmax, dmean, drange))
      self.normfacs[c] = {'scale':float(drange), 'offset':float(dmean)}
      self.data_train[:,:,idx] -= dmean
      self.data_train[:,:,idx] /= drange
      idx += 1
    print ('After Scaling:')
    print ('Column       Min       Max      Mean     Range     Scale    Offset')
    idx = 0
    for c in self.COL_U + self.COL_Y:
      dmean = np.mean (self.data_train[:,:,idx])
      dmax = np.max (self.data_train[:,:,idx])
      dmin = np.min (self.data_train[:,:,idx])
      drange = dmax - dmin
      print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, 
        dmin, dmax, dmean, drange, self.normfacs[c]['scale'], self.normfacs[c]['offset']))
      idx += 1
    fname = os.path.join(model_folder,'normfacs.json')
    fp = open (fname, 'w')
    json.dump (self.normfacs, fp, indent=2)
    fp.close()

  def loadNormalization(self, filename):
    fp = open (filename, 'r')
    self.normfacs = json.load (fp)
    fp.close()

  def loadAndApplyNormalization(self, filename):
    self.loadNormalization(filename)
    idx = 0
    for c in self.COL_U + self.COL_Y:
      dmean = self.normfacs[c]['offset']
      drange = self.normfacs[c]['scale']
      self.data_train[:,:,idx] -= dmean
      self.data_train[:,:,idx] /= drange
      idx += 1

  def initializeModelStructure(self):
    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    # structure is FHF cascade:
    #  inputs T, G, Fc, Ud, Vrms, G*Vrms, and ctl_mode as ub; outputs Vdc, Idc, Irms as y_hat
    self.F1 = MimoStaticNonLinearity(in_channels=len(idx_in), out_channels=len(idx_out), n_hidden=self.nh1, activation=self.activation)
    self.H1 = MimoLinearDynamicalOperator(in_channels=len(idx_out), out_channels=len(idx_out), n_b=self.nb, n_a=self.na, n_k=self.nk)
    self.y0 = torch.zeros((self.batch_size, self.na), dtype=torch.float)
    self.u0 = torch.zeros((self.batch_size, self.nb), dtype=torch.float)
    self.F2 = MimoStaticNonLinearity(in_channels=len(idx_out), out_channels=len(idx_out), n_hidden=self.nh2, activation=self.activation)

  def trainModelCoefficients(self):
    self.optimizer = torch.optim.Adam([
      {'params': self.F1.parameters(), 'lr': self.lr},
      {'params': self.H1.parameters(), 'lr': self.lr},
      {'params': self.F2.parameters(), 'lr': self.lr},
    ], lr=self.lr)
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)
    train_ds = PVInvDataset(self.data_train,in_size,out_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

    LOSS = []
    start_time = time.time()
    for itr in range(0, self.num_iter):
      for ub, yb in train_dl:
        self.optimizer.zero_grad()
        # Simulate FHF
        y_non = self.F1 (ub)
        y_lin = self.H1 (y_non, self.y0, self.u0)
        y_hat = self.F2 (y_lin)
        # Compute fit loss
        err_fit = yb - y_hat
        loss_fit = torch.sum(torch.abs(err_fit))
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % self.print_freq == 0:
          print('Iter {:4d} of {:4d} | Loss {:12.4f}'.format (itr, self.num_iter, loss_fit))

        # Optimize
        loss.backward()
        self.optimizer.step()
    train_time = time.time() - start_time
    return train_time, LOSS

  def saveModelCoefficients(self, model_folder):
    torch.save(self.F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
    torch.save(self.H1.state_dict(), os.path.join(model_folder, "H1.pkl"))
    torch.save(self.F2.state_dict(), os.path.join(model_folder, "F2.pkl"))

  def loadModelCoefficients(self, model_folder):
    B1 = torch.load(os.path.join(model_folder, "F1.pkl"))
    self.F1.load_state_dict(B1)
    B2 = torch.load(os.path.join(model_folder, "H1.pkl"))
    self.H1.load_state_dict(B2)
    B3 = torch.load(os.path.join(model_folder, "F2.pkl"))
    self.F2.load_state_dict(B3)

  def exportModel(self, filename):
    config = {'name':'PV1', 'type':'F1+H1+F2', 't_step': self.t_step}
    config['normfacs'] = {}
    for key, val in self.normfacs.items():
      config['normfacs'][key] = {'scale':val['scale'], 'offset':val['offset']}
    config['lr'] = self.lr
    config['num_iter'] = self.num_iter
    config['print_freq'] = self.print_freq
    config['batch_size'] = self.batch_size
    config['n_skip'] = self.n_skip
    config['n_trunc'] = self.n_trunc
    config['n_dec'] = self.n_dec
    config['na'] = self.na
    config['nb'] = self.nb
    config['nk'] = self.nk
    config['activation'] = self.activation
    config['nh1'] = self.nh1
    config['nh2'] = self.nh2
    config['COL_T'] = self.COL_T
    config['COL_Y'] = self.COL_Y
    if 'GVrms' not in self.COL_U:
      if 'GVrms' in self.normfacs:
        self.COL_U.append('GVrms')
    if 'Mode' not in self.COL_U:
      if 'Mode' in self.normfacs:
        self.COL_U.append('Mode')
    config['COL_U'] = self.COL_U

    self.append_lti (config, 'H1', self.H1)
    self.append_net (config, 'F1', self.F1)
    self.append_net (config, 'F2', self.F2)

    fp = open (filename, 'w')
    json.dump (config, fp, indent=2)
    fp.close()

  def testOneCase(self, case_idx):
    case_data = self.data_train[[case_idx],:,:]
    ub = torch.tensor (case_data[:,:,idx_in])
    y_non = self.F1 (ub)
    y_lin = self.H1 (y_non, self.y0, self.u0)
    y_hat = self.F2 (y_lin)
    print (ub.shape, y_non.shape, y_lin.shape, y_hat.shape)

    y_hat = y_hat.detach().numpy()[[0], :, :]
    y_true = np.transpose(case_data[0,:,idx_out])
    rmse = dynonet.metrics.error_rmse(y_true, y_hat[0])
    return rmse, y_hat, y_true, np.transpose(case_data[0,:,idx_in])

  def stepOneCase(self, case_idx):
    case_data = self.data_train[[case_idx],:,:]
    n = len(self.t)
    y_hat = np.zeros(shape=(n,len(idx_out)))
    ub = torch.zeros((1, 1, len(idx_in)), dtype=torch.float)
    print ('case_data', case_data.shape, 'y_hat', y_hat.shape)
    for i in range(n):
      u_row = case_data[:,i,idx_in]
#      print ('u_row', u_row.shape)
      ub[0,0,:] = torch.from_numpy(u_row[0,:])
#      print (i, 'u_row', u_row.shape, u_row)
#      print (i, 'ub', ub.shape, ub)
      y_non = self.F1 (ub)
#      print ('  y_non', y_non.shape)
      y_lin = self.H1 (y_non, self.y0, self.u0)
#      print ('  y_lin', y_lin.shape)
      y_row = self.F2 (y_lin)
#      print ('  y_row', y_row.shape)
      y_hat[i,:] = y_row.detach().numpy()

#    y_hat = y_hat.detach().numpy()[[0], :, :]
    y_true = np.transpose(case_data[0,:,idx_out])
    print ('rmse y_true', y_true.shape, 'y_hat', y_hat.shape)
    rmse = dynonet.metrics.error_rmse(y_true, y_hat)
    return rmse, y_hat, y_true, np.transpose(case_data[0,:,idx_in])

  def trainingErrors(self, bByCase=False):
    ub = torch.tensor (self.data_train[:,:,idx_in])
    y_non = self.F1 (ub)
    y_lin = self.H1 (y_non, self.y0, self.u0)
    y_hat = self.F2 (y_lin)

    y_hat = y_hat.detach().numpy()
    y_true = self.data_train[:,:,idx_out]
    self.n_cases = self.data_train.shape[0]

    icol = 0
    total_rmse = {}
    total_mae = {}
    if bByCase:
      case_rmse = lst1 = [dict() for i in range(self.n_cases)]
    else:
      case_rmse = None
    for col in self.COL_Y:
      SUMSQ = 0.0
      MAE = 0.0
      for icase in range(self.n_cases):
        y1 = y_true[icase,:,icol]
        y2 = y_hat[icase,:,icol]
        colmae = dynonet.metrics.error_mae(y1, y2)
        colrms = dynonet.metrics.error_rmse(y1, y2)
        if bByCase:
          case_rmse[icase][col] = colrms
        MAE += colmae
        SUMSQ += (colrms*colrms)
      MAE /= self.n_cases
      SUMSQ /= self.n_cases
      RMSE = math.sqrt(SUMSQ)
      total_mae[col] = MAE
      total_rmse[col] = RMSE
      icol += 1
    return total_rmse, total_mae, case_rmse
