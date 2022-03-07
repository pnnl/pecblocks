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
import control

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

class pv3():
  def __init__(self, training_config=None, sim_config=None):
    self.init_to_none()
    if training_config is not None:
      self.load_training_config (training_config)
    if sim_config is not None:
      self.load_sim_config (sim_config)

  def init_to_none(self):
    self.lr = None
    self.num_iter = None
    self.print_freq = None
    self.batch_size = None
    self.n_loss_skip = None
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

  def load_training_config(self, filename):
    fp = open (filename, 'r')
    config = json.load (fp)
    fp.close()
    self.lr = config['lr']
    self.num_iter = config['num_iter']
    self.print_freq = config['print_freq']
    self.batch_size = config['batch_size']
    self.n_skip = config['n_skip']
    self.n_trunc = config['n_trunc']
    self.n_loss_skip = config['n_loss_skip']
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
    self.t_step = 1.0e-3

  def read_lti(self, config):
    n_in = config['n_in']
    n_out = config['n_out']
    n_a = config['n_a']
    n_b = config['n_b']
    n_k = config['n_k']
    block = MimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b, n_a=n_a, n_k=n_k)
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
      self.set_idx_in_out()
      self.normfacs = config['normfacs']
      self.t_step = config['t_step']

  def load_sim_config(self, filename, model_only=True):
    fp = open (filename, 'r')
    config = json.load (fp)
    fp.close()
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

  def append_lti(self, model, label, H):
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
                                             self.n_dec, self.n_skip, self.n_trunc)
    print ('read', len(df_list), 'dataframes')

    # get the len of data of interest 
    df_1 = df_list[0]
    time_arr = np.array(df_1[self.COL_T], dtype=np.float32)
    self.t_step = np.mean(np.diff(time_arr.ravel())) #time_exp[1] - time_exp[0]
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
    cfg = json.load (fp)
    fp.close()
    if 'normfacs' in cfg:
      self.normfacs = cfg['normfacs']
    else:
      self.normfacs = cfg

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
    #  inputs COL_U as ub; outputs COL_Y as y_hat
    self.F1 = MimoStaticNonLinearity(in_channels=len(self.idx_in), out_channels=len(self.idx_out), n_hidden=self.nh1, activation=self.activation)
    self.H1 = MimoLinearDynamicalOperator(in_channels=len(self.idx_out), out_channels=len(self.idx_out), n_b=self.nb, n_a=self.na, n_k=self.nk)
    self.y0 = torch.zeros((self.batch_size, self.na), dtype=torch.float)
    self.u0 = torch.zeros((self.batch_size, self.nb), dtype=torch.float)
    self.F2 = MimoStaticNonLinearity(in_channels=len(self.idx_out), out_channels=len(self.idx_out), n_hidden=self.nh2, activation=self.activation)

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
        if self.n_loss_skip > 0:
          err_fit = yb[:,self.n_loss_skip:,:] - y_hat[:,self.n_loss_skip:,:]
        else:
          err_fit = yb - y_hat
#        loss_fit = torch.sum(torch.abs(err_fit))
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % self.print_freq == 0:
          print('Iter {:4d} of {:4d} | Loss {:12.6f}'.format (itr, self.num_iter, loss_fit))

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
    config['n_loss_skip'] = self.n_loss_skip
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

  def printStateDicts(self):
    print ('F1', self.F1.state_dict())
    print ('F2', self.F2.state_dict())
    print ('H1', self.H1.in_channels, self.H1.out_channels, self.H1.n_a, self.H1.n_b, self.H1.n_k)
    print (self.H1.state_dict())

  def testOneCase(self, case_idx):
    case_data = self.data_train[[case_idx],:,:]
    ub = torch.tensor (case_data[:,:,self.idx_in])
    y_non = self.F1 (ub)
    y_lin = self.H1 (y_non, self.y0, self.u0)
    y_hat = self.F2 (y_lin)
    print (ub.shape, y_non.shape, y_lin.shape, y_hat.shape)
#    self.printStateDicts()
#    print (y_lin)

    y_hat = y_hat.detach().numpy()[[0], :, :]
    y_true = np.transpose(case_data[0,:,self.idx_out])
    rmse = dynonet.metrics.error_rmse(y_true, y_hat[0])
    return rmse, y_hat, y_true, np.transpose(case_data[0,:,self.idx_in])

  def stepOneCase(self, case_idx):
    case_data = self.data_train[case_idx,:,:]
    n = len(self.t)
    y_hat = np.zeros(shape=(n,len(self.idx_out)))
    ub = torch.zeros((1, 1, len(self.idx_in)), dtype=torch.float)
    print ('case_data', case_data.shape, 'y_hat', y_hat.shape)
    self.start_simulation()
    for k in range(n):
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
        y_lin = torch.tensor (self.ysum, dtype=torch.float)
        y_hat[k,:] = self.F2(y_lin)
    return y_hat # the caller de_normalizes

  def trainingErrors(self, bByCase=False):
    ub = torch.tensor (self.data_train[:,:,self.idx_in])
    y_non = self.F1 (ub)
    y_lin = self.H1 (y_non, self.y0, self.u0)
    y_hat = self.F2 (y_lin)

    y_hat = y_hat.detach().numpy()
    y_true = self.data_train[:,:,self.idx_out]
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

  def set_LCL_filter(self, Lf, Cf, Lc):
    self.Lf = Lf
    self.Cf = Cf
    self.Lc = Lc

  def start_simulation(self):
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
#--------------------------------------------------------------
# implement a matrix of SISO transfer functions; y0 = u0*h00 + u1*h10 + u2*h20, etc.
#---------------------------------------------------------------------------------------
#    a_coeff = self.H1.a_coeff.detach().numpy()                                         
#    b_coeff = self.H1.b_coeff.detach().numpy()                                         
#    self.HTF = {}                                                                      
#    self.HTF_X0 = {}                                                                   
#    self.HTF_Y = np.zeros (self.H1.out_channels)                                       
#    for i in range(self.H1.in_channels):                                               
#      self.HTF[i] = {}                                                                 
#      self.HTF_X0[i] = {}                                                              
#      for j in range(self.H1.out_channels):                                            
#        self.HTF_X0[i][j] = 0.0                                                        
#        self.HTF[i][j] = control.TransferFunction (b_coeff[i, j, :],                   
#                                                   np.hstack(([1.0],a_coeff[i, j, :])),
#                                                   self.t_step)                        
#    for i in range(self.H1.in_channels):                                               
#      for j in range(self.H1.out_channels):                                            
#        print (self.HTF[i][j])                                                         
#---------------------------------------------------------------------------------------
# set up IIR filters for time step simulation
    self.a_coeff = self.H1.a_coeff.detach().numpy()                                         
    self.b_coeff = self.H1.b_coeff.detach().numpy()
    self.uhist = {}
    self.yhist = {}
    self.ysum = np.zeros(self.H1.out_channels)
    for i in range(self.H1.out_channels):
      self.uhist[i] = {}
      self.yhist[i] = {}
      for j in range(self.H1.in_channels):
        self.uhist[i][j] = np.zeros(self.H1.n_b)
        self.yhist[i][j] = np.zeros(self.H1.n_a)
                                             
# set up the static nonlinearity blocks for time step simulation                       
    self.F1.eval()
    self.F2.eval()

  def normalize (self, val, fac):
    return (val - fac['offset']) / fac['scale']

  def de_normalize (self, val, fac):
    return val * fac['scale'] + fac['offset']

  def step_simulation (self, G, T, Ud, Fc, Vrms, Mode, GVrms):
    Vc = np.complex (Vrms+0.0j)
    if self.Lf is not None:
      omega = 2.0*math.pi*Fc
      ZLf = np.complex(0.0+omega*self.Lf*1j)
      ZLc = np.complex(0.0+omega*self.Lc*1j)
      ZCf = np.complex(0.0-1j/omega/self.Cf)
    G = self.normalize (G, self.normfacs['G'])
    T = self.normalize (T, self.normfacs['T'])
    Ud = self.normalize (Ud, self.normfacs['Ud'])
    Fc = self.normalize (Fc, self.normfacs['Fc'])
    Vrms = self.normalize (Vrms, self.normfacs['Vrms'])
    Mode = self.normalize (Mode, self.normfacs['Mode'])
    GVrms = self.normalize (GVrms, self.normfacs['GVrms'])

    ub = torch.tensor ([T, G, Fc, Ud, Vrms, GVrms, Mode], dtype=torch.float)
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
      y_lin = torch.tensor (self.ysum, dtype=torch.float)
      y_hat = self.F2 (y_lin)

    Vdc = y_hat[0].item()
    Idc = y_hat[1].item()
    Irms = y_hat[2].item()

    Vdc = self.de_normalize (Vdc, self.normfacs['Vdc'])
    Idc = self.de_normalize (Idc, self.normfacs['Idc'])
    Irms = self.de_normalize (Irms, self.normfacs['Irms'])

    if self.Lf is not None:
      Ic = np.complex (Irms+0.0j)
      Vf = Vc + ZLc * Ic
      If = Vf / ZCf
      Is = Ic + If
      Vs = Vf + ZLf * Is
    else:
      Vs = Vc
      Is = np.complex (Irms+0.0j)

    return Vdc, Idc, Irms, Vs, Is

