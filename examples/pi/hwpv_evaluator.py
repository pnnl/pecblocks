# copyright 2021-2023 Battelle Memorial Institute
# HWPV model simulation code

import numpy as np

class model():
  def __init__(self):
    self.init_to_none()

  def init_to_none(self):
    self.COL_T = None
    self.COL_Y = None
    self.COL_U = None
    self.n_pad = None
    self.gtype = None
    self.na = None
    self.nb = None
    self.nk = None
    self.activation = None
    self.nh1 = None
    self.nh2 = None
    self.normfacs = None
    self.nin = None
    self.nout = None
    self.t_step = 1.0e-3

  def load_H_coefficients (self, H):
    a = np.zeros((H['n_out'], H['n_in'], H['n_a']))
    b = np.zeros((H['n_out'], H['n_in'], H['n_b']))
    for i in range (H['n_out']):
      for j in range (H['n_in']):
        a[i,j,:] = H['a_{:d}_{:d}'.format(i,j)]
        b[i,j,:] = H['b_{:d}_{:d}'.format(i,j)]
#    print ('\nHa\n', a)
#    print ('\nHb\n', b)
    return a, b

  def load_state_matrices (self, Q):
    A = np.zeros((Q['n_out'], Q['n_in'], Q['n_a'], Q['n_b'])) # TODO: n_a or n_b first?
    B = np.zeros((Q['n_out'], Q['n_in'], Q['n_a'])) # TODO: n_a or n_b?
    C = np.zeros((Q['n_out'], Q['n_in'], Q['n_b'])) # TODO: n_a or n_b?
    D = np.zeros((Q['n_out'], Q['n_in'])) # TODO: n_a or n_b?
    for i in range (Q['n_out']):
      for j in range (Q['n_in']):
        A[i,j,:,:] = Q['A_{:d}_{:d}'.format(i,j)]
        B[i,j,:] = Q['B_{:d}_{:d}'.format(i,j)]
        C[i,j:] = Q['C_{:d}_{:d}'.format(i,j)]
        D[i,j] = Q['D_{:d}_{:d}'.format(i,j)]
    return A, B, C, D

  def set_sim_config(self, config):
    self.name = config['name']
    self.blocks = config['type']
    self.normfacs = config['normfacs']
    self.COL_T = config['COL_T']
    self.COL_Y = config['COL_Y']
    self.COL_U = config['COL_U']
    self.na = config['na']
    self.nb = config['nb']
    self.nk = config['nk']
    self.activation = config['activation']
    self.nh1 = config['nh1']
    self.nh2 = config['nh2']
    self.t_step = config['t_step']
    if 'n_pad' in config:
      self.n_pad = config['n_pad']
    else:
      self.n_pad = 100
    if 'gtype' in config:
      self.gtype = config['gtype']
    else:
      self.gtype = 'iir'

    self.za, self.zb = self.load_H_coefficients (config['H1'])
    self.sa, self.sb = self.load_H_coefficients (config['H1s'])
    self.A, self.B, self.C, self.D = self.load_state_matrices (config['Q1s'])

    self.F1_n0w = np.array(config['F1']['net.0.weight'])
    self.F1_n0b = np.array(config['F1']['net.0.bias'])
    self.F1_n2w = np.array(config['F1']['net.2.weight'])
    self.F1_n2b = np.array(config['F1']['net.2.bias'])

    self.F2_n0w = np.array(config['F2']['net.0.weight'])
    self.F2_n0b = np.array(config['F2']['net.0.bias'])
    self.F2_n2w = np.array(config['F2']['net.2.weight'])
    self.F2_n2b = np.array(config['F2']['net.2.bias'])

    self.nout = len(self.F2_n2b)
    self.nin = self.F1_n0w.shape[1]

    print ('HWPV Model Structure (dt={:.6f}):'.format (self.t_step))
    print ('  Hza shape', self.za.shape)
    print ('  Hzb shape', self.zb.shape)
    print ('  Hsa shape', self.sa.shape)
    print ('  Hsb shape', self.sb.shape)
    print ('  QA shape', self.A.shape)
    print ('  QB shape', self.B.shape)
    print ('  QC shape', self.C.shape)
    print ('  QD shape', self.D.shape)
    print ('  F1 shapes 0w, 0b, 2w, 2b = ', self.F1_n0w.shape, self.F1_n0b.shape, 
           self.F1_n2w.shape, self.F1_n2b.shape, config['F1']['activation'])
    print ('  F2 shapes 0w, 0b, 2w, 2b = ', self.F2_n0w.shape, self.F2_n0b.shape, 
           self.F2_n2w.shape, self.F2_n2b.shape, config['F2']['activation'])
    print ('  {:d} inputs from {:s}'.format (self.nin, str(self.COL_U)))
    print ('  {:d} outputs from {:s}'.format (self.nout, str(self.COL_Y)))

  def start_simulation_z(self):
# set up IIR filters for time step simulation, nin == nout for H1
    self.uhist = {}
    self.yhist = {}
    self.ysum = np.zeros(self.nout)
    for i in range(self.nout):
      self.uhist[i] = {}
      self.yhist[i] = {}
      for j in range(self.nout):
        self.uhist[i][j] = np.zeros(self.nb)
        self.yhist[i][j] = np.zeros(self.na)
                                             
  def normalize (self, val, fac):
    return (val - fac['offset']) / fac['scale']

  def de_normalize (self, val, fac):
    return val * fac['scale'] + fac['offset']

  def tanh_layer (self, u, n0w, n0b, n2w, n2b):
    hidden = np.tanh (np.matmul(n0w, u) + n0b)
    output = np.matmul(n2w, hidden) + n2b
    return output

  def step_simulation_z (self, T, G, Fc, Md, Mq, Vrms, GVrms, Ctl):
    T = self.normalize (T, self.normfacs['T'])
    G = self.normalize (G, self.normfacs['G'])
    Fc = self.normalize (Fc, self.normfacs['Fc'])
    Md = self.normalize (Md, self.normfacs['Md'])
    Mq = self.normalize (Mq, self.normfacs['Mq'])
    Vrms = self.normalize (Vrms, self.normfacs['Vrms'])
    GVrms = self.normalize (GVrms, self.normfacs['GVrms'])
    Ctl = self.normalize (Ctl, self.normfacs['Ctl'])

    ub = np.array([T, G, Fc, Md, Mq, Vrms, GVrms, Ctl])
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    self.ysum[:] = 0.0
    for i in range(self.nout):
      for j in range(self.nout):
        uh = self.uhist[i][j]
        yh = self.yhist[i][j]
        uh[1:] = uh[:-1]
        uh[0] = y_non[j]
        ynew = np.sum(np.multiply(self.zb[i,j,:], uh)) - np.sum(np.multiply(self.za[i,j,:], yh))
        yh[1:] = yh[:-1]
        yh[0] = ynew
        self.ysum[i] += ynew
    y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)

    Vdc = self.de_normalize (y_hat[0], self.normfacs['Vdc'])
    Idc = self.de_normalize (y_hat[1], self.normfacs['Idc'])
    Id = self.de_normalize (y_hat[2], self.normfacs['Id'])
    Iq = self.de_normalize (y_hat[3], self.normfacs['Iq'])

    return Vdc, Idc, Id, Iq

  def start_simulation_sfe (self):
    self.q = np.zeros((self.nout, self.nout, self.nout))
    self.qdot = np.zeros(self.nout)
    self.ysum = np.zeros(self.nout)

  def step_simulation_sfe (self, T, G, Fc, Md, Mq, Vrms, GVrms, Ctl, h, log=False):
    T = self.normalize (T, self.normfacs['T'])
    G = self.normalize (G, self.normfacs['G'])
    Fc = self.normalize (Fc, self.normfacs['Fc'])
    Md = self.normalize (Md, self.normfacs['Md'])
    Mq = self.normalize (Mq, self.normfacs['Mq'])
    Vrms = self.normalize (Vrms, self.normfacs['Vrms'])
    GVrms = self.normalize (GVrms, self.normfacs['GVrms'])
    Ctl = self.normalize (Ctl, self.normfacs['Ctl'])

    ub = np.array([T, G, Fc, Md, Mq, Vrms, GVrms, Ctl])
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    if log:
      print ('\nub = {:s}'.format (str(ub)))
      print ('y_non = {:s}'.format (str(y_non)))
    self.ysum[:] = 0.0
    for i in range(self.nout):
      for j in range(self.nout):
        self.qdot = np.matmul (self.A[i,j], self.q[i,j]) + self.B[i,j] * y_non[j]
        self.q[i,j] += h * self.qdot
        self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
        if log:
          print ('  Q[{:d},{:d}] = {:s}'.format (i, j, str(self.q[i,j])))
    y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)

    Vdc = self.de_normalize (y_hat[0], self.normfacs['Vdc'])
    Idc = self.de_normalize (y_hat[1], self.normfacs['Idc'])
    Id = self.de_normalize (y_hat[2], self.normfacs['Id'])
    Iq = self.de_normalize (y_hat[3], self.normfacs['Iq'])
    if log:
      print ('Ysum = {:s}'.format(str(self.ysum)))
      print ('y_hat = {:s}'.format(str(y_hat)))
      print ('Vdc, Idc, Id, Iq = {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format (Vdc, Idc, Id, Iq))

    return Vdc, Idc, Id, Iq

  def start_simulation_sbe (self, h):
    self.q = np.zeros((self.nout, self.nout, self.nout))
    self.ysum = np.zeros(self.nout)
    self.lhs = {}
    for i in range(self.nout):
      self.lhs[i] = {}
      for j in range(self.nout):
        self.lhs[i,j] = np.linalg.cholesky(np.eye(self.nout) - h*self.A[i,j])

  def step_simulation_sbe (self, T, G, Fc, Md, Mq, Vrms, GVrms, Ctl, h, log=False):
    T = self.normalize (T, self.normfacs['T'])
    G = self.normalize (G, self.normfacs['G'])
    Fc = self.normalize (Fc, self.normfacs['Fc'])
    Md = self.normalize (Md, self.normfacs['Md'])
    Mq = self.normalize (Mq, self.normfacs['Mq'])
    Vrms = self.normalize (Vrms, self.normfacs['Vrms'])
    GVrms = self.normalize (GVrms, self.normfacs['GVrms'])
    Ctl = self.normalize (Ctl, self.normfacs['Ctl'])

    ub = np.array([T, G, Fc, Md, Mq, Vrms, GVrms, Ctl])
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    self.ysum[:] = 0.0
#    qold = np.copy(self.q)
    for i in range(self.nout):
      for j in range(self.nout):
        rhs = self.q[i,j] + self.B[i,j] * y_non[j] * h
        lhs = self.lhs[i,j]
        # forward substitutions
        for row in range(self.nout):
          for col in range(row):
            rhs[row] -= (lhs[row,col] * rhs[col])
          rhs[row] /= lhs[row,row]
        # back substitutions
        self.q[i,j,:] = 0.0
        for row in range(self.nout,0,-1):
          self.q[i,j,row-1] = (rhs[row-1] - np.dot(lhs[row:,row-1],self.q[i,j,row:])) / lhs[row-1,row-1]
        self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
    y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)
#    print ('max qdot', np.max(np.abs(qold - self.q)))

    Vdc = self.de_normalize (y_hat[0], self.normfacs['Vdc'])
    Idc = self.de_normalize (y_hat[1], self.normfacs['Idc'])
    Id = self.de_normalize (y_hat[2], self.normfacs['Id'])
    Iq = self.de_normalize (y_hat[3], self.normfacs['Iq'])
    if log:
      print ('Ysum = {:s}'.format(str(self.ysum)))
      print ('y_hat = {:s}'.format(str(y_hat)))
      print ('Vdc, Idc, Id, Iq = {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format (Vdc, Idc, Id, Iq))

    return Vdc, Idc, Id, Iq

