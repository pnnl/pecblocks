# copyright 2021-2024 Battelle Memorial Institute
# HWPV model simulation code

import numpy as np

LU_DECOMP = True

if LU_DECOMP:
  import scipy.linalg as la

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
    A = np.zeros((Q['n_out'], Q['n_in'], Q['n_a'], Q['n_a'])) # TODO: n_a or n_b first?
    B = np.zeros((Q['n_out'], Q['n_in'], Q['n_a'])) # TODO: n_a or n_b?
    C = np.zeros((Q['n_out'], Q['n_in'], Q['n_a'])) # TODO: n_a or n_b?
    D = np.zeros((Q['n_out'], Q['n_in'])) # TODO: n_a or n_b?
    for i in range (Q['n_out']):
      for j in range (Q['n_in']):
        A[i,j,:,:] = Q['A_{:d}_{:d}'.format(i,j)]
        B[i,j,:] = Q['B_{:d}_{:d}'.format(i,j)]
        C[i,j:] = Q['C_{:d}_{:d}'.format(i,j)]
        D[i,j] = Q['D_{:d}_{:d}'.format(i,j)]
    return A, B, C, D

  def set_sim_config(self, config, log=False):
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

    if log:
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

  def make_ub (self, inputs):
    ub = np.zeros(self.nin)
    for i in range(self.nin):
      ub[i] = self.normalize (inputs[i], self.normfacs[self.COL_U[i]])
    return ub

  def extract_y_hat(self, y_hat, log=False):
    ret = np.zeros(self.nout)
    for i in range(self.nout):
      ret[i] = self.de_normalize (y_hat[i], self.normfacs[self.COL_Y[i]])
    if log:
      print (self.COL_Y, '=', ','.join('{:.3f}'.format(ret[j]) for j in range(self.nout)))
    return ret

  def start_simulation_z(self, inputs):
# set up IIR filters for time step simulation, nin == nout for H1
    ub = self.make_ub (inputs)
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    self.uhist = {}
    self.yhist = {}
    self.ysum = np.zeros(self.nout)
    for i in range(self.nout):
      self.uhist[i] = {}
      self.yhist[i] = {}
      for j in range(self.nout):
        ynew = y_non[j] * np.sum(self.zb[i,j,:]) / (np.sum(self.za[i,j,:])+1.0)
        self.uhist[i][j] = np.ones(self.nb) * y_non[j]
        self.yhist[i][j] = np.ones(self.na) * ynew
        #print ('  start simulation', i, j, self.uhist[i][j], self.yhist[i][j])
                                             
  def normalize (self, val, fac):
    return (val - fac['offset']) / fac['scale']

  def de_normalize (self, val, fac):
    return val * fac['scale'] + fac['offset']

  def tanh_layer (self, u, n0w, n0b, n2w, n2b):
    hidden = np.tanh (np.matmul(n0w, u) + n0b)
    output = np.matmul(n2w, hidden) + n2b
    return output

  def step_simulation_z (self, inputs):
    ub = self.make_ub (inputs)
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
    return self.extract_y_hat (y_hat)

  def start_simulation_sfe (self, inputs, log=False):
    ub = self.make_ub (inputs)
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    rhs = -self.B * y_non
    lhs = np.linalg.solve (self.A, rhs)
    if log:
      print ('SFE IC ub\n', ub)
      print ('SFE IC y_non\n', y_non)
      print ('SFE IC RHS\n', rhs)
      print ('SFE IC LHS\n', lhs)

    self.q = np.zeros((self.nout, self.nout, self.nout))
    self.q = lhs
    self.qdot = np.zeros(self.nout)
    self.ysum = np.zeros(self.nout)
    if log:
      print ('SFE A\n', self.A)
      print ('SFE B\n', self.B)
      print ('SFE C\n', self.C)
      print ('SFE D\n', self.D)
      print ('SFE q\n', self.q)

  def step_simulation_sfe (self, inputs, h, log=False):
    ub = self.make_ub (inputs)
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    if log:
      print ('\nub = {:s}'.format (str(ub)))
      print ('y_non = {:s}'.format (str(y_non)))
    nsteps = max(1, int(h / self.t_step))
    for step in range(nsteps):
      self.ysum[:] = 0.0
      for i in range(self.nout):
        for j in range(self.nout):
          self.qdot = np.matmul (self.A[i,j], self.q[i,j]) + self.B[i,j] * y_non[j]
          self.q[i,j] += self.t_step * self.qdot
          self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
          if log:
            print ('  Q[{:d},{:d}] = {:s}'.format (i, j, str(self.q[i,j])))
    y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)
    if log:
      print ('Ysum = {:s}'.format(str(self.ysum)))
      print ('y_hat = {:s}'.format(str(y_hat)))
    return self.extract_y_hat (y_hat, log)

  def start_simulation_sbe (self, inputs, h, log=False):
    print ('SBE A {:s}\n'.format (str(self.A.shape)))#, self.A)
    print ('SBE B {:s}\n'.format (str(self.B.shape)))#, self.B)
    print ('SBE C {:s}\n'.format (str(self.C.shape)))#, self.C)
    print ('SBE D {:s}\n'.format (str(self.D.shape)))#, self.D)
    ub = self.make_ub (inputs)
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    print ('y_non {:s}\n'.format (str(y_non.shape)))#, y_non)
    self.q = np.linalg.solve (self.A, -self.B * y_non)
    self.ysum = np.zeros(self.nout)
    if log:
      print ('Backward Euler Method, LU Decomposition = {:s}, initial states:\n'.format (str(LU_DECOMP)), self.q)
      self.ysum[:] = 0.0
      for i in range(self.nout):
        for j in range(self.nout):
          self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
          self.ysum[i] += self.D[i,j] * y_non[j]
      y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)
      print ('Initial Outputs:', y_hat)
      print ('Denormalized:', self.extract_y_hat(y_hat))
#     print ('SBE A\n', self.A)
#     print ('SBE B\n', self.B)
#     print ('SBE C\n', self.C)
#     print ('SBE D\n', self.D)

    if LU_DECOMP:
      self.lu = {}
      self.piv = {}
      for i in range(self.nout):
        self.lu[i] = {}
        self.piv[i] = {}
        for j in range(self.nout):
          local_A = np.eye(self.nout) - h*self.A[i,j]
          self.lu[i,j], self.piv[i,j] = la.lu_factor(local_A)
    else:
      self.lhs = {}
      for i in range(self.nout):
        for j in range(self.nout):
          self.lhs[i,j] = np.eye(self.nout) - h*self.A[i,j]

    for k in range(200):
      yss = self.step_simulation_sbe (inputs, h)
    if log:
      print ('Padded Outputs:', yss)
      print ('Padded States:\n', self.q)


  def step_simulation_sbe (self, inputs, h, log=False):
    ub = self.make_ub (inputs)
    y_non = self.tanh_layer (ub, self.F1_n0w, self.F1_n0b, self.F1_n2w, self.F1_n2b)
    self.ysum[:] = 0.0
    if LU_DECOMP:
      for i in range(self.nout):
        for j in range(self.nout):
          rhs = self.q[i,j] + self.B[i,j] * y_non[j] * h
          self.q[i,j] = la.lu_solve ((self.lu[i,j], self.piv[i,j]), rhs)
          self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
          self.ysum[i] += self.D[i,j] * y_non[j]
    else:
      for i in range(self.nout):
        for j in range(self.nout):
          self.q[i,j] = np.linalg.solve (self.lhs[i,j], self.q[i,j] + self.B[i,j] * y_non[j] * h)
          self.ysum[i] += np.matmul (self.C[i,j], self.q[i,j])
          self.ysum[i] += self.D[i,j] * y_non[j]
    y_hat = self.tanh_layer (self.ysum, self.F2_n0w, self.F2_n0b, self.F2_n2w, self.F2_n2b)

    if log:
      print ('Ysum = {:s}'.format(str(self.ysum)))
      print ('y_hat = {:s}'.format(str(y_hat)))
    return self.extract_y_hat (y_hat, log)

