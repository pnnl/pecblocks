# copyright 2021-2024 Battelle Memorial Institute
# HW model training and simulation code for 3-phase inverters

import pandas as pd
import numpy as np
import os
import sys
import time
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.lti import MimoFirLinearDynamicalOperator
from dynonet.lti import StableSecondOrderMimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import dynonet.metrics
from pecblocks.common import PVInvDataset
import pecblocks.util
import json
import torch
import torch.nn as nn
import math
import control
import harold

class pv3():
  def __init__(self, training_config=None, sim_config=None):
    self.init_to_none()
    if training_config is not None:
      self.load_training_config (training_config)
    if sim_config is not None:
      self.load_sim_config (sim_config)

  def init_to_none(self):
    self.eps = 1e-8
    self.lr = None
    self.num_iter = None
    self.continue_iterations = False
    self.print_freq = None
    self.batch_size = None
    self.n_validation_pct = None
    self.n_validation_seed = None
    self.n_loss_skip = None
    self.n_pad = None
    self.gtype = None
    self.n_skip = None
    self.n_trunc = None
    self.n_dec = None
    self.na = None
    self.nb = None
    self.nk = None
    self.activation = None
    self.nh1 = None
    self.nh2 = None
    self.COL_T = None
    self.COL_Y = None
    self.COL_U = None
    self.idx_in = None
    self.idx_out = None
    self.data_train = None
    self.normfacs = None
    self.t = None
    self.n_cases = 0
    self.t_step = 1.0e-3
    self.Lf = None
    self.Lc = None
    self.Cf = None
    self.model_folder = None
    self.model_root = None
    self.data_path = None
    self.h5grp_prefix = None

  def load_training_config(self, filename):
    fp = open (filename, 'r')
    config = json.load (fp)
    fp.close()
    if 'model_folder' in config:
      self.model_folder = config['model_folder']
    else:
      self.model_folder = os.path.split(filename)[0]
    if 'model_root' in config:
      self.model_root = config['model_root']
    if 'data_path' in config:
      self.data_path = config['data_path']
    self.lr = config['lr']
    if 'eps' in config:
      self.eps = config['eps']
    self.num_iter = config['num_iter']
    if 'continue_iterations' in config:
      self.continue_iterations = config['continue_iterations']
    self.print_freq = config['print_freq']
    if 'h5grp_prefix' in config:
      self.h5grp_prefix = config['h5grp_prefix']
    if 't_step' in config:
      self.t_step = config['t_step']
    else:
      self.t_step = 1.0e-3
    self.batch_size = config['batch_size']
    if 'n_validation_pct' in config:
      self.n_validation_pct = config['n_validation_pct']
      self.n_validation_seed = config['n_validation_seed']
    self.n_skip = config['n_skip']
    self.n_trunc = config['n_trunc']
    self.n_loss_skip = config['n_loss_skip']
    if 'n_pad' in config:
      self.n_pad = config['n_pad']
    else:
      self.n_pad = 100
    if 'gtype' in config:
      self.gtype = config['gtype']
    else:
      self.gtype = 'iir'
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
    self.set_idx_in_out()

    self.data_train = None
    self.normfacs = None
    self.t = None
    self.n_cases = 0

  def make_mimo_block(self, gtype, n_in, n_out, n_a, n_b, n_k):
    print ('make_mimo_block', gtype)
    if gtype == 'fir':
      block = MimoFirLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b)
      if n_a > 0:
        print (' *** for FIR block, n_a should be 0, not', n_a)
    elif gtype == 'stable2nd':
      block = StableSecondOrderMimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out)
      if n_a != 2:
        print (' *** for stable 2nd-order block, n_a should be 2, not', n_a)
      if n_b != 3:
        print (' *** for stable 2nd-order block, n_b should be 3, not', n_b)
    elif gtype == 'iir':
      block = MimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b, n_a=n_a, n_k=n_k)
    else:
      print (' *** unrecognized gtype, using IIR')
      block = MimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b, n_a=n_a, n_k=n_k)
    return block

  def make_mimo_ylin(self, y_non):
    if self.gtype == 'iir':
      return self.H1 (y_non, self.y0, self.u0)
    elif self.gtype == 'fir':
      return self.H1 (y_non)
    elif self.gtype == 'stable2nd':
      return self.H1 (y_non, self.y0, self.u0)
    return None

  def read_lti(self, config):
    n_in = config['n_in']
    n_out = config['n_out']
    n_a = config['n_a']
    n_b = config['n_b']
    n_k = config['n_k']
    gtype = 'iir'
    block = self.make_mimo_block (gtype, n_in, n_out, n_a, n_b, n_k)
    dict = block.state_dict()
    for i in range(n_out):
      for j in range(n_in):
        a = config['a_{:d}_{:d}'.format(i, j)]
        b = config['b_{:d}_{:d}'.format(i, j)]
        dict['a_coeff'][i,j,:] = torch.Tensor(a)
        dict['b_coeff'][i,j,:] = torch.Tensor(b)

    block.load_state_dict (dict)
#    print ('state dict', block.state_dict())
    return block

  def read_net(self, config):
    n_in = config['n_in']
    n_out = config['n_out']
    n_hid = config['n_hid']
    actfun = config['activation']
    block = MimoStaticNonLinearity(in_channels=n_in, out_channels=n_out, n_hidden=n_hid, activation=actfun)
    dict = block.state_dict()
    for key in ['net.0.bias', 'net.0.weight', 'net.2.bias', 'net.2.weight']:
      dict[key] = torch.Tensor(np.array(config[key]))
    block.load_state_dict (dict)
#    print ('state dict', block.state_dict())
    return block

  def set_idx_in_out(self):
    self.idx_in = [0] * len(self.COL_U)
    self.idx_out = [0] * len(self.COL_Y)
    for i in range(len(self.COL_U)):
      self.idx_in[i] = i
    for i in range(len(self.COL_Y)):
      self.idx_out[i] = i + len(self.COL_U)
    print ('idx_in', self.idx_in)
    print ('idx_out', self.idx_out)

  def set_sim_config(self, config, model_only=True):
    self.name = config['name']
    self.blocks = config['type']
    self.H1 = self.read_lti(config['H1'])
    self.F1 = self.read_net(config['F1'])
    self.F2 = self.read_net(config['F2'])
    if not model_only:
      self.COL_T = config['COL_T']
      self.COL_Y = config['COL_Y']
      self.COL_U = config['COL_U']
      self.n_skip = config['n_skip']
      self.n_trunc = config['n_trunc']
      self.n_dec = config['n_dec']
      self.na = config['na']
      self.nb = config['nb']
      self.nk = config['nk']
      self.activation = config['activation']
      self.nh1 = config['nh1']
      self.nh2 = config['nh2']
      self.batch_size = config['batch_size']
      self.set_idx_in_out()
      self.normfacs = config['normfacs']
      self.t_step = config['t_step']
      if 'n_pad' in config:
        self.n_pad = config['n_pad']
      else:
        self.n_pad = 100
      if 'gtype' in config:
        self.gtype = config['gtype']
      else:
        self.gtype = 'iir'

  def load_sim_config(self, filename, model_only=True):
    fp = open (filename, 'r')
    config = json.load (fp)
    fp.close()
    self.model_folder = os.path.split(filename)[0]
    self.set_sim_config (config, model_only)

#-------------------------------------
#    print ('COL_U', self.COL_U)      
#    print ('COL_Y', self.COL_Y)      
#    print ('t_step', self.t_step)    
#    print ('F1', self.F1)            
#    print ('H1', self.H1)            
#    print ('F2', self.F2)            
#    print ('normfacs', self.normfacs)
#-------------------------------------

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

  def append_fir(self, model, label, H):
    n_in = H.in_channels
    n_out = H.out_channels
    model[label] = {'n_in': n_in, 'n_out': n_out, 'n_b': H.n_b, 'n_a': H.n_a, 'n_k':0}
    block = H.state_dict()
    b = block['G.weight'].numpy()
    for i in range(n_out):
      for j in range(n_in):
        key = 'b_{:d}_{:d}'.format(i, j)
        ary = b[i,j,:]
        model[label][key] = ary.tolist()

  def append_2nd(self, model, label, H):
    n_in = H.b_coeff.shape[1]
    n_out = H.b_coeff.shape[0]
    model[label] = {'n_in': n_in, 'n_out': n_out, 'n_b': 3, 'n_a': 2, 'n_k':0}
    block = H.state_dict()
    b = block['b_coeff'].numpy()
    # construct the a coefficients for IIR implementation
    rho = block['rho'].numpy().squeeze()
    psi = block['psi'].numpy().squeeze()
    r = 1 / (1 + np.exp(-rho))
    beta = np.pi / (1 + np.exp(-psi))
    a1 = -2 * r * np.cos(beta)
    a2 = r * r
    a = np.ones ((b.shape[0], b.shape[1], 2)) # don't write a0==1
    a[:,:,0] = a1
    a[:,:,1] = a2

    for i in range(n_out):
      for j in range(n_in):
        key = 'b_{:d}_{:d}'.format(i, j)
        ary = b[i,j,:]
        model[label][key] = ary.tolist()
        key = 'a_{:d}_{:d}'.format(i, j)
        ary = a[i,j,:]
        model[label][key] = ary.tolist()
        key = 'rho_{:d}_{:d}'.format(i, j)
        model[label][key] = float(rho[i,j])
        key = 'psi_{:d}_{:d}'.format(i, j)
        model[label][key] = float(psi[i,j])

  def append_lti(self, model, label, H):
    if self.gtype == 'fir':
      self.append_fir(model, label, H)
      return
    elif self.gtype == 'stable2nd':
      self.append_2nd(model, label, H)
      return
    n_in = H.in_channels
    n_out = H.out_channels
    model[label] = {'n_in': n_in, 'n_out': n_out, 'n_b': H.n_b, 'n_a': H.n_a, 'n_k': H.n_k}
    block = H.state_dict()
    a = block['a_coeff']
    b = block['b_coeff']
  #  print ('a_coeff shape:', a.shape) # should be (n_out, n_in, n_a==n_b)
    for i in range(n_out):
      for j in range(n_in):
        key = 'a_{:d}_{:d}'.format(i, j)
        ary = a[i,j,:].numpy()
        model[label][key] = ary.tolist()

        key = 'b_{:d}_{:d}'.format(i, j)
        ary = b[i,j,:].numpy()
        model[label][key] = ary.tolist()

  def loadTrainingData(self, data_path):
    df_list = pecblocks.util.read_hdf5_file (data_path, self.COL_T + self.COL_Y + self.COL_U, 
                                             self.n_dec, self.n_skip, self.n_trunc, prefix=self.h5grp_prefix)
    print ('read', len(df_list), 'dataframes')

    # get the len of data of interest 
    df_1 = df_list[0]
    time_arr = np.array(df_1[self.COL_T], dtype=np.float32)
    new_t_step = np.mean(np.diff(time_arr.ravel())) #time_exp[1] - time_exp[0]
    if abs(self.t_step - new_t_step)/self.t_step > 0.001:
      print ('*** warning: specified t_step {:.6f} differs from data {:.6f}'.format (self.t_step, new_t_step))
    self.t_step = new_t_step
    data_len = df_1[self.COL_T].size
    self.t = np.arange(data_len)*self.t_step

    n_input_output = len(self.COL_U) + len(self.COL_Y)

    # organize the training dataset by scenario, time and features
    self.n_cases = len(df_list)
    data_mat = np.empty((self.n_cases,data_len,n_input_output))
    print ('dt={:.6f} data_len={:d} n_io={:d} n_case={:d}'.format (self.t_step, data_len, n_input_output, self.n_cases))
    for sc_idx in range(self.n_cases):
      df_data = df_list[sc_idx][self.COL_U+self.COL_Y]
      data_mat[sc_idx,:,:] = np.array(df_data)
    self.data_train = data_mat.astype(np.float32)
    print (self.COL_U, self.COL_Y, self.data_train.shape)

  def applyAndSaveNormalization(self):
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
      # dmean = 0.5 * (dmax + dmin) # middle of the range over scenarios and time
      dmean = np.mean (self.data_train[:,:,idx]) # mean over scenarios and time
      drange = dmax - dmin
      if abs(drange) <= 0.0:
        drange = 1.0
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
      if abs(drange) <= 0.0:
        drange = 1.0
      print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, 
        dmin, dmax, dmean, drange, self.normfacs[c]['scale'], self.normfacs[c]['offset']))
      idx += 1
    fname = os.path.join(self.model_folder,'normfacs.json')
    fp = open (fname, 'w')
    json.dump (self.normfacs, fp, indent=2)
    fp.close()

  def loadNormalization(self, filename=None):
    if filename is None:
      filename = os.path.join(self.model_folder,'normfacs.json')
    fp = open (filename, 'r')
    cfg = json.load (fp)
    fp.close()
    if 'normfacs' in cfg:
      self.normfacs = cfg['normfacs']
    else:
      self.normfacs = cfg

  def loadAndApplyNormalization(self, filename=None, bSummary=False):
    if bSummary:
      print ('Before Scaling:')
      print ('Column       Min       Max      Mean     Range')
      idx = 0
      for c in self.COL_U + self.COL_Y:
        dmax = np.max (self.data_train[:,:,idx])
        dmin = np.min (self.data_train[:,:,idx])
        dmean = np.mean (self.data_train[:,:,idx]) # mean over scenarios and time
        drange = dmax - dmin
        idx += 1
        print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, dmin, dmax, dmean, drange))
    self.loadNormalization(filename)
    self.applyNormalization()
    if bSummary:
      idx = 0
      print ('After Scaling:')
      print ('Column       Min       Max      Mean     Range     Scale    Offset')
      for c in self.COL_U + self.COL_Y:
        dmean = np.mean (self.data_train[:,:,idx])
        dmax = np.max (self.data_train[:,:,idx])
        dmin = np.min (self.data_train[:,:,idx])
        drange = dmax - dmin
        idx += 1
        print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, 
          dmin, dmax, dmean, drange, self.normfacs[c]['scale'], self.normfacs[c]['offset']))

  def applyNormalization(self):
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
    #  inputs COL_U as ub; outputs COL_Y as y_hat
    self.F1 = MimoStaticNonLinearity(in_channels=len(self.idx_in), out_channels=len(self.idx_out), n_hidden=self.nh1, activation=self.activation)
    self.H1 = self.make_mimo_block(gtype=self.gtype, n_in=len(self.idx_out), n_out=len(self.idx_out), n_b=self.nb, n_a=self.na, n_k=self.nk)
    if self.gtype in ['iir', 'stable2nd']:
      self.y0 = torch.zeros((self.batch_size, self.na), dtype=torch.float)
      self.u0 = torch.zeros((self.batch_size, self.nb), dtype=torch.float)
    else:
      self.y0 = None
      self.u0 = None
    self.F2 = MimoStaticNonLinearity(in_channels=len(self.idx_out), out_channels=len(self.idx_out), n_hidden=self.nh2, activation=self.activation)

  def trainModelCoefficients(self, bMAE = False):
    self.optimizer = torch.optim.Adam([
      {'params': self.F1.parameters(), 'lr': self.lr},
      {'params': self.H1.parameters(), 'lr': self.lr},
      {'params': self.F2.parameters(), 'lr': self.lr},
    ], lr=self.lr, eps=self.eps)
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)

    # split the data into training and validation datasets
    total_ds = PVInvDataset (self.data_train, in_size, out_size)
    nvalidation = int (len(total_ds) * float (self.n_validation_pct) / 100.0)
    ntraining = len(total_ds) - nvalidation
    splits = [ntraining, nvalidation]
    train_ds, valid_ds = torch.utils.data.random_split (total_ds, splits, 
                                                        generator=torch.Generator().manual_seed(self.n_validation_seed))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
    validation_scale = float(len(train_ds)) / float(len(valid_ds))
    print ('Dataset split:', len(total_ds), len(train_ds), len (valid_ds), 'validation_scale={:.3f}'.format(validation_scale))

    if self.continue_iterations:
      print ('continuing iterations on existing model coefficients')
      self.loadModelCoefficients()

    LOSS = []
    VALID = []
    lossfile = os.path.join(self.model_folder, "Loss.npy")
    start_time = time.time()
    for itr in range(0, self.num_iter):
      epoch_loss = 0.0
      self.F1.train()
      self.H1.train()
      self.F2.train()
      for ub, yb in train_dl: # batch loop
        self.optimizer.zero_grad()
        # Simulate FHF
        y_non = self.F1 (ub)
        y_lin = self.make_mimo_ylin (y_non)
        y_hat = self.F2 (y_lin)
        # Compute fit loss
        if self.n_loss_skip > 0:
          err_fit = yb[:,self.n_loss_skip:,:] - y_hat[:,self.n_loss_skip:,:]
        else:
          err_fit = yb - y_hat
        if bMAE:
          loss_fit = torch.sum(torch.abs(err_fit))
        else:
          loss_fit = torch.mean(err_fit**2)
        loss = loss_fit
        # Optimize on this batch
        loss.backward()
        self.optimizer.step()
#        print ('  batch size={:d} loss={:12.6f}'.format (ub.shape[0], loss_fit))
        epoch_loss += loss_fit

      LOSS.append(epoch_loss.item())

      # validation loss
      valid_loss = 0.0
      self.F1.eval()
      self.H1.eval()
      self.F2.eval()
      for ub, yb in valid_dl:
        y_non = self.F1 (ub)
        y_lin = self.make_mimo_ylin (y_non)
        y_hat = self.F2 (y_lin)
        # Compute fit loss
        if self.n_loss_skip > 0:
          err_fit = yb[:,self.n_loss_skip:,:] - y_hat[:,self.n_loss_skip:,:]
        else:
          err_fit = yb - y_hat
        if bMAE:
          loss_fit = torch.sum(torch.abs(err_fit))
        else:
          loss_fit = torch.mean(err_fit**2)
        valid_loss += loss_fit * validation_scale

      VALID.append(valid_loss.item())
      if itr % self.print_freq == 0:
        print('Epoch {:4d} of {:4d} | Training Loss {:12.6f} | Validation Loss {:12.6f}'.format (itr, self.num_iter, epoch_loss, valid_loss))
        self.saveModelCoefficients()
        np.save (lossfile, [LOSS, VALID])

    train_time = time.time() - start_time
    np.save (lossfile, [LOSS, VALID])
    return train_time, LOSS, VALID

  def saveModelCoefficients(self):
    torch.save(self.F1.state_dict(), os.path.join(self.model_folder, "F1.pkl"))
    torch.save(self.H1.state_dict(), os.path.join(self.model_folder, "H1.pkl"))
    torch.save(self.F2.state_dict(), os.path.join(self.model_folder, "F2.pkl"))

  def loadModelCoefficients(self):
    B1 = torch.load(os.path.join(self.model_folder, "F1.pkl"))
    self.F1.load_state_dict(B1)
    B2 = torch.load(os.path.join(self.model_folder, "H1.pkl"))
    self.H1.load_state_dict(B2)
#   print (self.H1)
#   print ('b_coeff', self.H1.b_coeff)
#   print ('rho', self.H1.rho)
#   print ('psi', self.H1.psi)
    B3 = torch.load(os.path.join(self.model_folder, "F2.pkl"))
    self.F2.load_state_dict(B3)

  def make_H1Q1s(self, Hz):
    if self.gtype != 'iir':
      return None, None
    n_in = Hz.in_channels
    n_out = Hz.out_channels
    n_a = Hz.n_a
    n_b = Hz.n_b
    H1s = {'n_in':n_in, 'n_out':n_out, 'n_a': n_a+1, 'n_b': n_b}
    Q1s = {'n_in':n_in, 'n_out':n_out, 'n_a': n_a, 'n_b': n_b}
    btf, atf = Hz.get_tfdata()
    b_coeff, a_coeff = Hz.__get_ba_coeff__()
#   print ('btf', btf.shape, btf[0][0])
#   print ('b_coeff', b_coeff.shape, b_coeff[0][0])
#   print ('atf', atf.shape, atf[0][0])
#   print ('a_coeff', a_coeff.shape, a_coeff[0][0])
    a = atf
    b = b_coeff

    # convert each MIMO channel one at a time
    for i in range(n_out):
      for j in range(n_in):
        Hz_har = harold.Transfer (b[i][j], a[i][j], dt=self.t_step)
        Hs_har = harold.undiscretize (Hz_har, method='forward euler', prewarp_at=499.99, q='none')
        numHs = np.array(Hs_har.num).ravel().tolist()
        denHs = np.array(Hs_har.den).ravel().tolist()
        Hs = control.TransferFunction(numHs, denHs)
        poles = control.pole(Hs)
        real_poles = np.real(poles)
        imag_poles = np.imag(poles) / (2.0 * np.pi)
        if np.all(real_poles < 0):
          flag = ''
        else:
          flag = '** unstable **'
        frequencies_present = []
        for hz in imag_poles:
          if hz > 0.0:
            frequencies_present.append (hz)
        print ('H1s[{:d}][{:d}] {:s} Real Poles:'.format(i, j, flag), real_poles, 'Freqs [Hz]:', frequencies_present)
        H1s['b_{:d}_{:d}'.format(i,j)] = np.array(Hs.num).squeeze().tolist()
        H1s['a_{:d}_{:d}'.format(i,j)] = np.array(Hs.den).squeeze().tolist()
        Qs = control.tf2ss (Hs)
        Q1s['A_{:d}_{:d}'.format(i,j)] = np.array(Qs.A).squeeze().tolist()
        Q1s['B_{:d}_{:d}'.format(i,j)] = np.array(Qs.B).squeeze().tolist()
        Q1s['C_{:d}_{:d}'.format(i,j)] = np.array(Qs.C).squeeze().tolist()
        Q1s['D_{:d}_{:d}'.format(i,j)] = np.array(Qs.D).squeeze().tolist()

    return H1s, Q1s

  def exportModel(self, filename):
    config = {'name':'PV3', 'type':'F1+H1+F2', 't_step': self.t_step}
    config['normfacs'] = {}
    for key, val in self.normfacs.items():
      config['normfacs'][key] = {'scale':val['scale'], 'offset':val['offset']}
    config['model_folder'] = self.model_folder
    config['model_root'] = self.model_root
    config['data_path'] = self.data_path
    config['lr'] = self.lr
    config['eps'] = self.eps
    config['h5grp_prefix'] = self.h5grp_prefix
    config['num_iter'] = self.num_iter
    config['continue_iterations'] = self.continue_iterations
    config['print_freq'] = self.print_freq
    config['batch_size'] = self.batch_size
    config['n_validation_pct'] = self.n_validation_pct
    config['n_validation_seed'] = self.n_validation_seed
    config['n_skip'] = self.n_skip
    config['n_trunc'] = self.n_trunc
    config['n_dec'] = self.n_dec
    config['n_loss_skip'] = self.n_loss_skip
    config['n_pad'] = self.n_pad
    config['gtype'] = self.gtype
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
    config['H1s'], config['Q1s'] = self.make_H1Q1s(self.H1)
    self.append_net (config, 'F1', self.F1)
    self.append_net (config, 'F2', self.F2)

    fp = open (filename, 'w')
    json.dump (config, fp, indent=2)
    fp.close()

  def printStateDicts(self):
    print ('F1', self.F1.state_dict())
    print ('F2', self.F2.state_dict())
    print ('H1', self.H1.in_channels, self.H1.out_channels, self.H1.n_a, self.H1.n_b, self.H1.n_k)
    print (self.H1.state_dict())

  def testOneCase(self, case_idx, npad):
    case_data = self.data_train[[case_idx],:,:]
    udata = case_data[0,:,self.idx_in].squeeze().transpose()

    # padding initial conditions
    ic = np.zeros((npad, len(self.idx_in)))
    for i in range(len(self.idx_in)):
      ic[:,i] = udata[0,i]
#    print (case_data.shape, udata.shape, ic.shape)
#    print (ic)
    udata = np.concatenate ((ic, udata))

    ub = torch.tensor (np.expand_dims (udata, axis=0), dtype=torch.float)
    y_non = self.F1 (ub)
    y_lin = self.make_mimo_ylin (y_non)
    y_hat = self.F2 (y_lin)
#    print (ub.shape, y_non.shape, y_lin.shape, y_hat.shape)
#    self.printStateDicts()
#    print (y_lin)

    y_hat = y_hat.detach().numpy()[[0], npad:, :].squeeze()
    y_true = np.transpose(case_data[0,:,self.idx_out]).squeeze()
    rmse = dynonet.metrics.error_rmse(y_true, y_hat)
    mae = dynonet.metrics.error_mae(y_true, y_hat)
    udata = udata[npad:,:]
#    print (rmse.shape, mae.shape, y_hat.shape, y_true.shape, udata.shape)
    return rmse, mae, y_hat, y_true, udata

  def simulateVectors(self, T, G, Fc, Md, Mq, Vrms, GVrms, Ctl, npad):
    T = self.normalize (T, self.normfacs['T'])
    G = self.normalize (G, self.normfacs['G'])
    Fc = self.normalize (Fc, self.normfacs['Fc'])
    Md = self.normalize (Md, self.normfacs['Md'])
    Mq = self.normalize (Mq, self.normfacs['Mq'])
    Vrms = self.normalize (Vrms, self.normfacs['Vrms'])
    GVrms = self.normalize (GVrms, self.normfacs['GVrms'])
    Ctl = self.normalize (Ctl, self.normfacs['Ctl'])

    data = np.array([T, G, Fc, Md, Mq, Vrms, GVrms, Ctl]).transpose()
#   print (data.shape)
#   # padding initial conditions
    ic = np.zeros((npad, 8))
    ic[:,0] = T[0]
    ic[:,1] = G[0]
    ic[:,2] = Fc[0]
    ic[:,3] = Md[0]
    ic[:,4] = Mq[0]
    ic[:,5] = Vrms[0]
    ic[:,6] = GVrms[0]
    ic[:,7] = Ctl[0]
    data = np.concatenate ((ic, data))
#    print (data)
    ub = torch.tensor (np.expand_dims(data, axis=0), dtype=torch.float)
#    print ('ub, y0, u0 shapes =', ub.shape, self.y0.shape, self.u0.shape)

    y_non = self.F1 (ub)
    y_lin = self.make_mimo_ylin (y_non)
    y_hat = self.F2 (y_lin)

#   print ('y_non', y_non.shape)
#   print (y_non[0,0,:])
#   print ('y_lin', y_lin.shape)
#   print (y_lin[0,0:10,:])
#   print ('y_hat', y_hat.shape)
#   print (y_hat[0,0:10,:])

    y_hat = y_hat.detach().numpy()[[0], :, :].squeeze()

    Vdc = self.de_normalize (y_hat[npad:,0], self.normfacs['Vdc'])
    Idc = self.de_normalize (y_hat[npad:,1], self.normfacs['Idc'])
    Id = self.de_normalize (y_hat[npad:,2], self.normfacs['Id'])
    Iq = self.de_normalize (y_hat[npad:,3], self.normfacs['Iq'])
    return Vdc, Idc, Id, Iq

  def stepOneCase(self, case_idx, npad):
    case_data = self.data_train[case_idx,:,:]
    n = len(self.t)
    y_hat = np.zeros(shape=(n,len(self.idx_out)))
    ub = torch.zeros((1, 1, len(self.idx_in)), dtype=torch.float)
    self.start_simulation()
    ub = torch.tensor (case_data[0,self.idx_in]) # initial inputs
    for k in range(-npad, n):
      if k > 0:
        ub = torch.tensor (case_data[k,self.idx_in])
      with torch.no_grad():
        y_non = self.F1 (ub)
        self.ysum[:] = 0.0
        for i in range(self.H1.out_channels):
          for j in range(self.H1.in_channels):
            uh = self.uhist[i][j]
            yh = self.yhist[i][j]
            uh[1:] = uh[:-1]
            uh[0] = y_non[j]
            ynew = np.sum(np.multiply(self.b_coeff[i,j,:], uh)) - np.sum(np.multiply(self.a_coeff[i,j,:], yh))
            yh[1:] = yh[:-1]
            yh[0] = ynew
            self.ysum[i] += ynew
        if k >= 0:
          y_lin = torch.tensor (self.ysum, dtype=torch.float)
          y_hat[k,:] = self.F2(y_lin)
    y_true = case_data[:,self.idx_out].squeeze()
    udata = case_data[:,self.idx_in].squeeze()
#    print ('shapes: case_data', case_data.shape, 'y_hat', y_hat.shape, 'y_true', y_true.shape, 'udata', udata.shape)
    rmse = dynonet.metrics.error_rmse(y_true, y_hat)
    mae = dynonet.metrics.error_mae(y_true, y_hat)
    return rmse, mae, y_hat, y_true, udata

  def trainingErrors(self, bByCase=False):
    self.n_cases = self.data_train.shape[0]
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)
    total_rmse = np.zeros(out_size)
    total_mae = np.zeros(out_size)
    if bByCase:
      case_rmse = np.zeros([self.n_cases, out_size])
      case_mae = np.zeros([self.n_cases, out_size])
    else:
      case_rmse = None
      case_mae = None

    total_ds = PVInvDataset (self.data_train, in_size, out_size)
    total_dl = torch.utils.data.DataLoader(total_ds, batch_size=self.batch_size, shuffle=False)

    icase = 0
    for ub, y_true in total_dl: # batch loop
      y_non = self.F1 (ub)
      y_lin = self.make_mimo_ylin (y_non)
      y_hat = self.F2 (y_lin)
      y1 = y_true.detach().numpy()
      y2 = y_hat.detach().numpy()
      y_err = np.abs(y1-y2)
      y_sqr = y_err*y_err
      nb = y_err.shape[0]
      npts = y_err.shape[1]
      ncol = y_err.shape[2]
      mae = np.mean (y_err, axis=1) # nb x ncol
      mse = np.mean (y_sqr, axis=1)
      total_mae += np.sum(mae, axis=0)
      total_rmse += np.sum(mse, axis=0)
      if bByCase:
        iend = icase + nb
        case_mae[icase:iend,:] = mae[:,:]
        case_rmse[icase:iend,:] = mse[:,:]
        icase = iend
    total_rmse = np.sqrt(total_rmse / self.n_cases)
    total_mae /= self.n_cases
    if bByCase:
      case_rmse = np.sqrt(case_rmse)
    return total_rmse, total_mae, case_rmse, case_mae

  def set_LCL_filter(self, Lf, Cf, Lc):
    self.Lf = Lf
    self.Cf = Cf
    self.Lc = Lc

  def check_poles(self):
    #----------------------------------------------
    # create a matrix of SISO transfer functions
    #----------------------------------------------
    HTF = {}
    n_in = 1
    n_out = 1
    if self.gtype == 'fir':
      print ('FIR H(z) is always stable')
      return
    elif self.gtype == 'stable2nd':
      b = self.H1.b_coeff.detach().numpy().squeeze()
      n_in = b.shape[1]
      n_out = b.shape[0]
      rho = self.H1.rho.detach().numpy().squeeze()
      psi = self.H1.psi.detach().numpy().squeeze()
#     print ('b', b.shape, b)
#     print ('rho', rho.shape, rho)
#     print ('psi', psi.shape, psi)
      r = 1 / (1 + np.exp(-rho))
      beta = np.pi / (1 + np.exp(-psi))
      a1 = -2 * r * np.cos(beta)
      a2 = r * r
#     print ('a1', a1)
#     print ('a2', a2)
      a = np.ones ((b.shape[0], b.shape[1], 3))
      a[:,:,1] = a1
      a[:,:,2] = a2
#     print ('a', a)
      for i in range(n_out):
        HTF[i] = {}
        for j in range(n_in):
          HTF[i][j] = control.TransferFunction (b[i, j, :], a[i, j, :], self.t_step)
    elif self.gtype == 'iir':
      a_coeff = self.H1.a_coeff.detach().numpy()
      b_coeff = self.H1.b_coeff.detach().numpy()
      n_in = b_coeff.shape[1]
      n_out = b_coeff.shape[0]
      num, den = self.H1.get_tfdata()
#     print ('num', num.shape, num)
#     print ('den', den.shape, den)
#     print ('n_k', self.H1.n_k, self.H1.n_a, self.H1.n_b)
      G_tf = control.TransferFunction (num[0][0], den[0][0], self.t_step)
#     print ('G_tf', G_tf)
      for i in range(n_out):
        HTF[i] = {}
        for j in range(n_in):
          HTF[i][j] = control.TransferFunction (b_coeff[i, j, :], # indexing ::-1 will reverse the order
                                                np.hstack(([1.0],a_coeff[i, j, :])),
                                                self.t_step)
#     print ('HTF', HTF[0][0])
    else:
      print ('cannot check poles for unknown gtype', self.gtype)
      return

    for i in range(n_out):
      for j in range(n_in):
        flag = ''
        polemag = abs(HTF[i][j].poles())
        if np.any(polemag >= 1.0):
          print ('==H(z)[{:d}][{:d}] ({:s} from {:s})'.format(i, j, self.COL_Y[i], self.COL_Y[j]))
          print (HTF[i][j])
          flag = '*** UNSTABLE ***'
          print (' {:s} Magnitudes of Poles: {:s}'.format (flag, str(polemag)))
        else:
          pass
#          print ('==H(z)[{:d}][{:d}] ({:s} from {:s}) pole magnitudes {:s}'.format(i, j, self.COL_Y[i], self.COL_Y[j], str(polemag)))

  def start_simulation(self, bPrint=False):
#-------------------------------------------------------------------------------------------------------
## control MIMO TransferFunction, needs 'slycot' package for forced_response
#    a_coeff = self.H1.a_coeff.detach().numpy()                                                         
#    b_coeff = self.H1.b_coeff.detach().numpy()                                                         
#    a_poly = np.empty_like(a_coeff, shape=(self.H1.out_channels, self.H1.in_channels, self.H1.n_a + 1))
#    a_poly[:, :, 0] = 1                                                                                
#    a_poly[:, :, 1:] = a_coeff[:, :, :]                                                                
#    b_poly = np.array(b_coeff)                                                                         
#    self.H1_sys = control.TransferFunction(b_poly, a_poly, self.t_step)                                
#    self.H1_t = [self.t_step]                                                                          
#    self.H1_x0 = [0.0, 0.0, 0.0]                                                                       
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------- 
## dynoNet implementation, wrong dimensionality for loop
#    self.y0 = torch.zeros((1, self.H1.n_a), dtype=torch.float)
#    self.u0 = torch.zeros((1, self.H1.n_b), dtype=torch.float)
#    self.H1.eval()                                            
#---------------------------------------------------------------------------------------
# set up IIR filters for time step simulation
    self.b_coeff = self.H1.b_coeff.detach().numpy()
    if bPrint:
      print ('start_simulation [n_a, n_b, n_in, n_out]=[{:d} {:d} {:d} {:d}]'.format (self.H1.n_a, self.H1.n_b, self.H1.in_channels, self.H1.out_channels))
    if self.gtype == 'stable2nd' and not hasattr(self.H1, 'a_coeff'):
      rho = self.H1.rho.detach().numpy().squeeze()
      psi = self.H1.psi.detach().numpy().squeeze()
      r = 1 / (1 + np.exp(-rho))
      beta = np.pi / (1 + np.exp(-psi))
      a1 = -2 * r * np.cos(beta)
      a2 = r * r
      self.a_coeff = np.ones ((self.b_coeff.shape[0], self.b_coeff.shape[1], 2)) #  3))
      self.a_coeff[:,:,0] = a1
      self.a_coeff[:,:,1] = a2
      if bPrint:
        print ('constructed a_coeff')                                         
    else:
      self.a_coeff = self.H1.a_coeff.detach().numpy()
      if bPrint:
        print ('existing a_coeff')
    if bPrint:
      print (self.a_coeff)
      print (self.b_coeff)
    self.uhist = {}
    self.yhist = {}
    self.ysum = np.zeros(self.H1.out_channels)
    for i in range(self.H1.out_channels):
      self.uhist[i] = {}
      self.yhist[i] = {}
      for j in range(self.H1.in_channels):
        self.uhist[i][j] = np.zeros(self.H1.n_b)
        self.yhist[i][j] = np.zeros(self.H1.n_a)
    return self.COL_U
                                             
# set up the static nonlinearity blocks for time step simulation                       
    self.F1.eval()
    self.F2.eval()

  def normalize (self, val, fac):
    return (val - fac['offset']) / fac['scale']

  def de_normalize (self, val, fac):
    return val * fac['scale'] + fac['offset']

  def step_simulation (self, vals, nsteps=1):
#   Vc = np.complex (Vrms+0.0j)
#   if self.Lf is not None:
#     omega = 2.0*math.pi*Fc
#     ZLf = np.complex(0.0+omega*self.Lf*1j)
#     ZLc = np.complex(0.0+omega*self.Lc*1j)
#     ZCf = np.complex(0.0-1j/omega/self.Cf)
    for i in range(len(vals)):
      vals[i] = self.normalize (vals[i], self.normfacs[self.COL_U[i]])

    ub = torch.tensor (vals, dtype=torch.float)
    with torch.no_grad():
      y_non = self.F1 (ub)
      for iter in range(nsteps):
        self.ysum[:] = 0.0
        for i in range(self.H1.out_channels):
          for j in range(self.H1.in_channels):
            uh = self.uhist[i][j]
            yh = self.yhist[i][j]
            uh[1:] = uh[:-1]
            uh[0] = y_non[j]
            ynew = np.sum(np.multiply(self.b_coeff[i,j,:], uh)) - np.sum(np.multiply(self.a_coeff[i,j,:], yh))
            yh[1:] = yh[:-1]
            yh[0] = ynew
            self.ysum[i] += ynew
      y_lin = torch.tensor (self.ysum, dtype=torch.float)
      y_hat = self.F2 (y_lin)

    if len(y_hat) < 4:
      Idc = y_hat[0].item()
      Id = y_hat[1].item()
      Iq = y_hat[2].item()
      Idc = self.de_normalize (Idc, self.normfacs['Idc'])
      Id = self.de_normalize (Id, self.normfacs['Id'])
      Iq = self.de_normalize (Iq, self.normfacs['Iq'])
      return Idc, Id, Iq

    Vdc = y_hat[0].item()
    Idc = y_hat[1].item()
    Id = y_hat[2].item()
    Iq = y_hat[3].item()

    Vdc = self.de_normalize (Vdc, self.normfacs['Vdc'])
    Idc = self.de_normalize (Idc, self.normfacs['Idc'])
    Id = self.de_normalize (Id, self.normfacs['Id'])
    Iq = self.de_normalize (Iq, self.normfacs['Iq'])

#   if self.Lf is not None:
#     Ic = np.complex (Irms+0.0j)
#     Vf = Vc + ZLc * Ic
#     If = Vf / ZCf
#     Is = Ic + If
#     Vs = Vf + ZLf * Is
#   else:
#     Vs = Vc
#     Is = np.complex (Irms+0.0j)

    return Vdc, Idc, Id, Iq

