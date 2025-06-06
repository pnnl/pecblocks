# copyright 2021-2025 Battelle Memorial Institute
# HW model training and simulation code for 3-phase inverters

"""
  The **pv3_poly** class supports training and evaluation of generalized
  block diagram models of three-phase and single-phase inverters.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.lti import MimoFirLinearDynamicalOperator
from dynonet.lti import StableSecondOrderMimoLinearDynamicalOperator
from dynonet.lti import StableSecondOrderMimoLinearDynamicalOperatorX
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
  """Implementation of HWPV model training, export, and forward evaluation.

  This class is configured from a JSON file. The normal usage is to load a 
  training dataset and normalize it. Then perform one or more of the 
  following operations: 

  - Train the model
  - Export the model
  - Evaluate model metrics
  - Forward evaluation, or simulation

  Attributes:
    eps (float): numerical stability parameter for the Adam optimizer in PyTorch
    lr (float): learning rate for PyTorch
    num_iter (int): number of iterations (epochs) to perform in the next training run
    continue_iterations (bool): if *True*, start the next set of training epochs from the existing model saved in *pkl* files. If *False*, reset the initial model to randomized parameters.
    print_freq (int): epoch interval for printing status information during model training
    batch_size (int): the data loader's batch size for training in PyTorch
    n_validation_pct (int): percentage of cases in *data_train* to reserve for validation loss
    n_validation_seed (int): a random number to choose the cases reserved for validation loss
    n_loss_skip (int): number of decimated-in-time points to exclude from loss evaluation during training, used to exclude the initialization transients.
    n_pad (int): number of decimated-in-time points to pre-pad the training data with *t=0* initial values, used to mitigate *dynoNet*'s behavior of always starting H1 from rest. Due to bias coefficients from F1, the initial H1 inputs are generally non-zero.
    gtype (str): type of H1 block, may be *iir*, *fir*, *stable2ndx*. (*stable2nd* is deprecated because it does not allow distinct real poles, only complex conjugates.)
    n_skip (int): number of decimated-in-time points to exclude from the beginning of each event in *data_train*
    n_trunc (int): number of decimated-in-time points to exclude from the end of each event in *data_train*
    n_dec (int): decimation-in-time interval for *data_train*, e.g., 1 for every point, 5 for every fifth point. Decimation may have already been done in preparing the input data for the HDF5 file.
    na (int): number of learnable denominator coefficients in H1, not including a0=1
    nb (int): number of learnable numerator coefficients in H1
    nk (int): number of delay cells in H1 (never tested in pecblocks)
    activation (str): activation function for the F1 and F2 blocks, may be *tanh* (preferred), *sigmoid* or *relu*
    nh1 (int): number of hidden cells in F1
    nh2 (int): number of hidden cells in F2
    COL_T (list(str)): time channel name of *data_train*; should be an array of length 1
    COL_U (list(str)): input channel names of *data_train*
    COL_Y (list(str)): output channel names of *data_train*
    idx_in (list(int)): channel indices of *data_train* corresponding to *COL_U*
    idx_out (list(int)): channel indices of *data_train* corresponding to *COL_Y*
    data_train (array_like(float, ndim=3)): three-dimensional Numpy array of training data loaded from an HDF5 file. First dimension is the case number, second dimension is the time point index, third dimension is the channel index.
    normfacs (dict): the min, max, mean, offset, and scaling factor for each channel in *data_train*
    t (float): the series of time points in *data_train*
    n_cases (int): number of cases (or events) in *data_train*
    t_step (float): the decimated time step in *data_train*
    Lf (float): inverter's output filter inductance, before *Cf*
    Lc (float): inverter's output filter inductance, at the PCC
    Cf (float): inverter's output filter capacitance
    model_folder (str): folder to save trained and exported model data
    model_root (str): base name for the exported model
    data_path (str): path and file name to the hdf5 input data for *data_train*
    h5grp_prefix (str): prefix for the 0-based indexing of cases in *data_train*
    clamps (dict): configuration for clamping losses
    sensitivity (dict): configuration for dV/dI or dI/dV sensitivity losses
    d_key (str): automatically determined as *Id* for a Norton model or *Vd* for a Thevenin model
    q_key (str): automatically determined as *Iq* for a Norton model or *Vq* for a Thevenin model
    sens_counter (int): tracks the number of recursive function calls to build sensitivity evaluation sets
  """
  def __init__(self, training_config=None, sim_config=None):
    """Constructor method. If called with no parameters, then *load_training_config* or *load_sim_config* must be called later before use.

    See Also:
      :func:`load_training_config`
      :func:`load_sim_config`

    Args:
      training_config (str): JSON file name with a configuration for model training.
      sim_config (str): JSON file name with an exported configuration for model evaluation. These file nams generally end in *_fhf.json*
    """
    self.init_to_none()
    if training_config is not None:
      self.load_training_config (training_config)
    if sim_config is not None:
      self.load_sim_config (sim_config)

  def init_to_none(self):
    """Initializes class attributes to a default value, usually *None*
    """
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
    self.clamps = None
    self.sensitivity = None
    self.d_key = None
    self.q_key = None
    self.sens_counter = 0

  def load_training_config(self, filename):
    """Loads a configuration set for model training.

    The configuration may include optional *sensitivity* and *clamps* members
    for enhanced loss evaluations. The JSON file may include other members
    that will be ignored here, e.g., differently named *data_path* attributes
    from differeng computing platforms. For another example, a *sensitivity* member
    could be renamed as *disable_sensitivity*, which has the effect of excluding
    *sensitivity* from the loss function without discarding its configuration data
    from the JSON file. This function does not load exported model parameters
    from the JSON file, but these parameters may be available from existing *pkl*
    files in the *model_folder* 

    Args:
      filename(str): JSON file name with the configuration parameters
    """
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
    if 'clamps' in config:
      self.clamps = config['clamps']
    if 'sensitivity' in config:
      self.sensitivity = config['sensitivity']
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
    """Create the H1 block of a requested type and dimension.

    The multiple-input multiple-output block will comprise a matrix of
    discrete H(z) transfer functions between each input and output pair,
    each having the same order.

    Args:
      gtype(str): use *stable2ndx* for stable 2nd-order, *iir* for infinite impulse response, or *fir* for finite impulse response
      n_in(int): number of input channels (F1 outputs)
      n_out(int): number of output channels (F2 inputs)
      n_a(int): number of learnable denominator coefficients for each H(z)
      n_b(int): number of learnable numerator coefficients for each H(z)
      n_k(int): number of delay cells for each H(z)

    Returns:
      MimoLinearDynamicalOperator: a subclass from *dynoNet*
    """
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
    elif gtype == 'stable2ndx':
      block = StableSecondOrderMimoLinearDynamicalOperatorX(in_channels=n_in, out_channels=n_out)
      if n_a != 2:
        print (' *** for stable 2nd-order extended block, n_a should be 2, not', n_a)
      if n_b != 3:
        print (' *** for stable 2nd-order extended block, n_b should be 3, not', n_b)
    elif gtype == 'iir':
      block = MimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b, n_a=n_a, n_k=n_k)
    else:
      print (' *** unrecognized gtype, using IIR')
      block = MimoLinearDynamicalOperator(in_channels=n_in, out_channels=n_out, n_b=n_b, n_a=n_a, n_k=n_k)
    return block

  def make_mimo_ylin(self, y_non):
    """Evaluates *H1* from the *F1* output, used in training or evaluation. The *dynoNet* function signature depends on *gtype*.

    Args:
      y_non(MimoStaticNonLinearity): F1 evaluated

    Returns:
      MimoLinearDynamicalOperator: H1 evaluated
    """
    if self.gtype == 'iir':
      return self.H1 (y_non, self.y0, self.u0)
    elif self.gtype == 'fir':
      return self.H1 (y_non)
    elif self.gtype == 'stable2nd':
      return self.H1 (y_non, self.y0, self.u0)
    elif self.gtype == 'stable2ndx':
      return self.H1 (y_non, self.y0, self.u0)
    return None

  def read_lti(self, config):
    """Creates *H1* from the JSON configuration

    Args:
      config(dict): configuration section for *H1* from JSON file, which must include the exported model members

    Returns:
      MimoLinearDynamicalOperator: *H1* of the correct subclass
    """
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
    """Creates *F1* or *F2* from the JSON configuration

    Args:
      config(dict): configuration section for *F1* or *F2* from JSON file, which must include the exported model members

    Returns:
      MimoStaticNonLinearity: *F1* or *F2*
    """
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
    """Populates *idx_in*, *idx_out*, *d_key*, and *q_key*.

    The indices are set sequentially for the channels as ordered in *COL_U* and *COL_Y*.
    If *COL_Y* includes *Id* and *Iq*, *d_key = Id* and *q_key = Iq* for a Norton model.
    If *COL_Y* includes *Vd* and *Vq*, *d_key = Vd* and *q_key = Vq* for a Thevenin model.
    """
    self.idx_in = [0] * len(self.COL_U)
    self.idx_out = [0] * len(self.COL_Y)
    for i in range(len(self.COL_U)):
      self.idx_in[i] = i
    for i in range(len(self.COL_Y)):
      self.idx_out[i] = i + len(self.COL_U)
    print ('idx_in', self.idx_in)
    print ('idx_out', self.idx_out)
    if 'Id' in self.COL_Y:
      self.d_key = 'Id'
    elif 'Vd' in self.COL_Y:
      self.d_key = 'Vd'
    if 'Iq' in self.COL_Y:
      self.q_key = 'Iq'
    elif 'Vq' in self.COL_Y:
      self.q_key = 'Vq'

  def set_sim_config(self, config, model_only=True):
    """Loads the attributes for simulation by forward evaluation

    Args:
      config(dict): configuration from JSON file, which must include the exported model parameters
      model_only(bool): if *True*, don't load the original model training configuration. If *False*, load configuration parameters that are sufficient for testing against the original *data_train*
    """
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
    """Loads a reduced configuration set for model evaluation.

    Only the *name*, *type*, *H1*, *F1*, and *F2* members from model export
    are essential.  If available, other configuration members will be included.

    Args:
      filename(str): JSON file name with the exported model configuration parameters. It will generally end with *_fhf.json*.
    """
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
    """Adds *F1* or *F2* to the model export.

    Args:
      model(dict): configuration that will be exported to a JSON file
      label(str): either 'F1' or 'F2' to label the block
      F(MimoStaticNonLinearity): either *F1* or *F2* containing the parameters
    """
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
    """Adds *H1* as a finite impulse response block to the model export.

    Args:
      model(dict): configuration that will be exported to a JSON file
      label(str): should be 'H1' to label the block
      H(MimoFirLinearDynamicalOperator): *H1* containing the parameters
    """
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
    """Adds *H1* as a stable 2nd-order block to the model export, limited to complex conjugate poles.

    Args:
      model(dict): configuration that will be exported to a JSON file
      label(str): should be 'H1' to label the block
      H(StableSecondOrderMimoLinearDynamicalOperator): *H1* containing the parameters
    """
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

  def append_2ndx(self, model, label, H):
    """Adds *H1* as a stable 2nd-order block to the model export, allows distinct real and complex conjugate poles.

    Args:
      model(dict): configuration that will be exported to a JSON file
      label(str): should be 'H1' to label the block
      H(StableSecondOrderMimoLinearDynamicalOperatorX): *H1* containing the parameters
    """
    n_in = H.b_coeff.shape[1]
    n_out = H.b_coeff.shape[0]
    model[label] = {'n_in': n_in, 'n_out': n_out, 'n_b': 3, 'n_a': 2, 'n_k':0}
    block = H.state_dict()
    b = block['b_coeff'].numpy()
    # construct the a coefficients for IIR implementation
    alpha1 = block['alpha1'].numpy().squeeze()
    alpha2 = block['alpha2'].numpy().squeeze()
    a1 = 2.0 * np.tanh(alpha1)
    a1abs = np.abs(a1)
    a2 = a1abs + (2.0 - a1abs) * 1.0/(1+np.exp(-alpha2)) - 1.0
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
        key = 'alpha1_{:d}_{:d}'.format(i, j)
        model[label][key] = float(alpha1[i,j])
        key = 'alpha2_{:d}_{:d}'.format(i, j)
        model[label][key] = float(alpha2[i,j])

  def append_lti(self, model, label, H):
    """Adds *H1* to the model export. 

    Checks *gtype* and if not FIR or stable 2nd-order, treat as IIR.

    Args:
      model(dict): configuration that will be exported to a JSON file
      label(str): should be 'H1' to label the block
      H(MimoLinearDynamicalOperator): *H1* containing the IIR parameters
    """
    if self.gtype == 'fir':
      self.append_fir(model, label, H)
      return
    elif self.gtype == 'stable2nd':
      self.append_2nd(model, label, H)
      return
    elif self.gtype == 'stable2ndx':
      self.append_2ndx(model, label, H)
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
    """Read the HDF5 training records into a NumPy array.

    Args:
      data_path(str): path and file name to the HDF5 data file.

    Yields:
      data_train loaded as a 3-dimensional NumPy array.
    """
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
    """Scans *data_train* to identify the range of each channel, then normalize them.

    Yields:

      - Each channel normalized to vary from 0.0 to 1.0 over all cases and times.
      - Normalization factors and ranges saved in *normfacs*
    """
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
      self.normfacs[c] = {'scale':float(drange), 'offset':float(dmean), 'max':float(dmax), 'min':float(dmin)}
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
    """Retrieves the saved normalization facgtors

    Args:
      filename(str): name of JSON file with a 'normfacs' section. If *None*, try 'normfacs.json'

    Yields:
      *normfacs* will be updated but not applied to the data.
    """
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
    """Reads *normfacs* from a JSON file, then applies them to *data_train*

    Args:
      filename(str): name of JSON file with a 'normfacs' section. If *None*, try 'normfacs.json'
      bSummary(bool): if *True*, print a summary of the unnormalized and normalized data

    Yields:
      *normfacs* will be updated and applied to *data_train*.
    """
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
    """Normalizes *data_train* using *normfacs*
    """
    idx = 0
    for c in self.COL_U + self.COL_Y:
      dmean = self.normfacs[c]['offset']
      drange = self.normfacs[c]['scale']
      self.data_train[:,:,idx] -= dmean
      self.data_train[:,:,idx] /= drange
      idx += 1

  def initializeModelStructure(self):
    """Set the blocks up for training or forward evaluation.

    Yields:
      *F1*, *H1*, and *F2* constructed to match numbers of channels and *gtype*, random initial coefficients.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    # structure is FHF cascade:
    #  inputs COL_U as ub; outputs COL_Y as y_hat
    self.F1 = MimoStaticNonLinearity(in_channels=len(self.idx_in), out_channels=len(self.idx_out), n_hidden=self.nh1, activation=self.activation)
    self.H1 = self.make_mimo_block(gtype=self.gtype, n_in=len(self.idx_out), n_out=len(self.idx_out), n_b=self.nb, n_a=self.na, n_k=self.nk)
    if self.gtype in ['iir', 'stable2nd', 'stable2ndx']:
      self.y0 = torch.zeros((self.batch_size, self.na), dtype=torch.float)
      self.u0 = torch.zeros((self.batch_size, self.nb), dtype=torch.float)
    else:
      self.y0 = None
      self.u0 = None
    self.F2 = MimoStaticNonLinearity(in_channels=len(self.idx_out), out_channels=len(self.idx_out), n_hidden=self.nh2, activation=self.activation)

  def find_sens_channel_indices (self, targets, available):
    """Pick out the training dataset channel numbers used in sensitivity evaluations. Call separately for the input channels and the output channesl. *Internal*

    Args:
      targets (list(str)): array of channel names used in the sensitivity evaluation set
      available (list(str)): array of channel names available in a model's training dataset

    Returns:
      int: length of the next return array, equal to *len(targets)*
      list(int): array of training dataset channel numbers
    """
    n = len(targets)
    idx = []
    for i in range(n):
      idx.append (available.index(targets[i]))
    return n, idx

  def build_sens_baselines (self, bases, step_vals, cfg, keys, indices, lens, level):
    """Recursive function to add a set of operating points to the sensitivity evaluation set. Uses a depth-first approach. When the last channel number is processed, the recursion will back up to a previous channel number that was not fully processed yet. *Internal*

    Args:
      bases (list(float)[]): array of *step_vals* for operating points in the sensitivity evaluation set 
      step_vals (list(float)): input channel values for a *pv3_poly* model steady-state operating point 
      cfg (dict): the *sensitivity* member of a *pv3_poly* configuration, which incluces a member *sets* contained keyed channel names
      keys (list(str)): list of channel names from the *sets* member of *cfg*, each of these corresponds to a *level* of recursion
      indices (list(int)): keeps track of the channel number to resume processing whenever *level* reachs the last *key* 
      lens (list(int)): the number of operating point values for each named channel in *keys*
      level (int): enters with 0, backs up at the length of *keys* minus 1

    Yields:
      Appending to *bases*. Updates *sens_counter* in each call.

    Returns:
      None
    """
    self.sens_counter += 1

    key = keys[level]
    ary = cfg['sets'][key]
    idx = cfg['idx_set'][key]

#    print ('baseline', self.sens_counter, level, key, idx, step_vals)

    if level+1 == len(keys): # add basecases at the lowest level
      for i in range(lens[level]):
        step_vals[idx] = ary[i]
        bases.append (step_vals.copy())
    else: # propagate this new value down to lower levels
      step_vals[idx] = ary[indices[level]]

    if level+1 < len(keys):
      level += 1
      self.build_sens_baselines (bases, step_vals, cfg, keys, indices, lens, level)
    else:
      level -= 1
      while level >= 0:
        if indices[level]+1 >= lens[level]:
          level -= 1
        else:
          indices[level] += 1
          indices[level+1:] = 0
          self.build_sens_baselines (bases, step_vals, cfg, keys, indices, lens, level)

  def setup_sensitivity_losses(self):
    """Parses the *sensitivity* configuration data, constructs the operating points for the
    sensitivity evaluation set. The existence of *sensitivity* should be checked before calling.
    """
    self.sensitivity['n_in'], self.sensitivity['idx_in'] = self.find_sens_channel_indices (self.sensitivity['inputs'], self.COL_U)
    self.sensitivity['n_out'], self.sensitivity['idx_out'] = self.find_sens_channel_indices (self.sensitivity['outputs'], self.COL_Y)
    self.sensitivity['idx_set'] = {}
    for key in self.sensitivity['sets']:
      self.sensitivity['idx_set'][key] = self.COL_U.index(key)
    print ('inputs', self.sensitivity['inputs'], self.sensitivity['idx_in'])
    print ('outputs', self.sensitivity['outputs'], self.sensitivity['idx_out'])
    print ('sets', self.sensitivity['idx_set'])
    if 'GVrms' in self.sensitivity:
      self.sensitivity['idx_g_rms'] = self.COL_U.index (self.sensitivity['GVrms']['G'])
      self.sensitivity['idx_d_rms'] = self.COL_U.index (self.sensitivity['GVrms']['Vd'])
      self.sensitivity['idx_q_rms'] = self.COL_U.index (self.sensitivity['GVrms']['Vq'])
      self.sensitivity['idx_gdqrms'] = self.COL_U.index ('GVrms')
      self.sensitivity['k'] = self.sensitivity['GVrms']['k']
    elif 'GIrms' in self.sensitivity:
      self.sensitivity['idx_g_rms'] = self.COL_U.index (self.sensitivity['GIrms']['G'])
      self.sensitivity['idx_d_rms'] = self.COL_U.index (self.sensitivity['GIrms']['Id'])
      self.sensitivity['idx_q_rms'] = self.COL_U.index (self.sensitivity['GIrms']['Iq'])
      self.sensitivity['idx_gdqrms'] = self.COL_U.index ('GIrms')
      self.sensitivity['k'] = self.sensitivity['GIrms']['k']
    else:
      self.sensitivity['idx_g_rms'] = None
      self.sensitivity['idx_d_in'] = self.COL_U.index (self.sensitivity['inputs'][0])
      self.sensitivity['idx_q_in'] = self.COL_U.index (self.sensitivity['inputs'][1])
    if self.sensitivity['idx_g_rms'] is not None:
      print ('GDQrms', self.sensitivity['idx_g_rms'], self.sensitivity['idx_d_rms'], self.sensitivity['idx_q_rms'], 
        self.sensitivity['k'], self.sensitivity['idx_gdqrms'])
    else:
      print ('DQ indices (no G)', self.sensitivity['idx_d_in'], self.sensitivity['idx_q_in'])

    self.sens_bases = []
    vals = np.zeros (len(self.COL_U))
    keys = list(self.sensitivity['sets'])
    indices = np.zeros(len(keys), dtype=int)
    lens = np.zeros(len(keys), dtype=int)
    for i in range(len(keys)):
      lens[i] = len(self.sensitivity['sets'][keys[i]])
    self.sens_counter = 0
#    print (keys, indices, lens)
    self.build_sens_baselines (self.sens_bases, vals, self.sensitivity, keys, indices, lens, 0)
    print (len(self.sens_bases), 'sensitivity base cases constructed in', self.sens_counter, 'function calls')

  def sensitivity_response (self, vals, bPrint=False):
    """Calculate the maximum d-axis and q-axis model sensitivity, using pre-calculated *sens_mat* for *H1*

    Args:
      vals (list(float)): input vector for *ub*

    Returns:
      float: maximum d-axis sensitivity, de-normalized
      float: maximum q-axis sensitivity, de-normalized
    """
    for i in range(len(vals)):
      vals[i] = self.normalize (vals[i], self.normfacs[self.COL_U[i]])

    ub = torch.tensor (vals, dtype=torch.float) # requires_grad=False
    y_non = self.F1 (ub)
    ysum = torch.sum (y_non * self.sens_mat, dim=1)
    y_hat = self.F2 (ysum)
    if bPrint:
      print ('sens resp', y_hat)
    ACd = self.de_normalize (y_hat[self.sensitivity['idx_out'][0]], self.normfacs[self.d_key])
    ACq = self.de_normalize (y_hat[self.sensitivity['idx_out'][1]], self.normfacs[self.q_key])
    return ACd, ACq

  # coefficients as trainable tensors instead of numpy, use self.H1.b as they are, but pad self.H1.a with ones
  def start_sensitivity_simulation(self, bPrint=False):
    """Build *sens_mat* from the *H1* coefficients with z=-1, for efficiency in the sensitivity evaluations.

    Args:
      bPrint (bool): if *True*, print out the *H1* coefficients and *sens_mat*
    """
    if bPrint:
      print ('  start_sensitivity_simulation [n_a, n_b, n_in, n_out]=[{:d} {:d} {:d} {:d}]'.format (self.H1.n_a, self.H1.n_b, self.H1.in_channels, self.H1.out_channels))
    if self.gtype == 'stable2nd' and not hasattr(self.H1, 'a_coeff'): 
      r = 1 / (1 + torch.exp(-self.H1.rho))
      beta = np.pi / (1 + torch.exp(-self.H1.psi))
      a1 = -2 * r * torch.cos(beta)
      a2 = r * r
      self.H1.a_coeff = torch.cat ((a1, a2), dim=2)
      if bPrint:
        print ('    constructed a_coeff', self.H1.a_coeff.shape, self.H1.a_coeff)
        print ('    beta', beta.shape, beta)
        print ('    a1', a1.shape, a1)
        print ('    a2', a2.shape, a2)
    if self.gtype == 'stable2ndx' and not hasattr(self.H1, 'a_coeff'): 
      a1 = 2.0 * torch.tanh(self.H1.alpha1)
      a1abs = torch.abs(a1)
      a2 = a1abs + (2.0 - a1abs) * torch.sigmoid(self.H1.alpha2) - 1.0
      self.H1.a_coeff = torch.cat ((a1, a2), dim=2)
      if bPrint:
        print ('    constructed a_coeff', self.H1.a_coeff.shape, self.H1.a_coeff)
        print ('    a1', a1.shape, a1)
        print ('    a2', a2.shape, a2)
    else:
      if bPrint:
        print ('    existing a_coeff')
    sa_coeff = torch.cat ((self.H1.a_coeff, torch.ones ((self.H1.b_coeff.shape[0], self.H1.b_coeff.shape[1], 1))), dim=2)
    sum_b = torch.sum(self.H1.b_coeff, dim=2)
    sum_sa = torch.sum(sa_coeff, dim=2)
    self.sens_mat = torch.div(sum_b, sum_sa)
    if bPrint:
      print ('    a=\n', self.H1.a_coeff)
      print ('   sa=\n', sa_coeff)
      print ('    b=\n', self.H1.b_coeff)
      print ('  mat=\n', self.sens_mat)

  def calc_sensitivity_losses(self, bPrint=False):
    """Calculate a sensitivity loss term that can be added to the fitting loss.

    The torch functions are used to facilitate optimization during training.
    This function is called after the fitting of each batch, because the model coefficients were updated.
    It is not called for each dataset, because the sensitivity only depends on model coefficients.

    Args:
      bPrint (bool): if *True*, print some diagnostics

    Returns:
      float: weighted sensitivity loss; add this to the fitting loss
      float: maximum sensitivity, normalized
    """
    delta = self.sensitivity['delta']
    idx_g = self.sensitivity['idx_g_rms']
    if idx_g is not None:
      idx_d = self.sensitivity['idx_d_rms']
      idx_q = self.sensitivity['idx_q_rms']
      idx_gdqrms = self.sensitivity['idx_gdqrms']
      krms = self.sensitivity['k']
    else:
      idx_d = self.sensitivity['idx_d_in']
      idx_q = self.sensitivity['idx_q_in']
    max_sens = torch.tensor (0.0, requires_grad=True)
    sens_floor = torch.tensor (1.0e-8, requires_grad=True)
    self.start_sensitivity_simulation (bPrint=bPrint)

    for vals in self.sens_bases:
      Ud0 = vals[idx_d]
      Uq0 = vals[idx_q]
      Ud1 = Ud0 + delta
      Uq1 = Uq0 + delta

      if idx_g is not None:
        vals[idx_gdqrms] = vals[idx_g] * krms * math.sqrt (Ud0*Ud0 + Uq0*Uq0)
      Yd0, Yq0 = self.sensitivity_response (vals.copy(), bPrint=False)

      if idx_g is not None:
        vals[idx_gdqrms] = vals[idx_g] * krms * math.sqrt (Ud1*Ud1 + Uq0*Uq0)
      vals[idx_d] = Ud1
      vals[idx_q] = Uq0
      Yd1, Yq1 = self.sensitivity_response (vals.copy(), bPrint=False)

      if idx_g is not None:
        vals[idx_gdqrms] = vals[idx_g] * krms * math.sqrt (Ud0*Ud0 + Uq1*Uq1)
      vals[idx_d] = Ud0
      vals[idx_q] = Uq1
      Yd2, Yq2 = self.sensitivity_response (vals.copy(), bPrint=False)
      #prevent aliasing the base cases
      vals[idx_q] = Uq0

      sens = torch.max (torch.stack ((torch.abs(Yd1 - Yd0), torch.abs(Yq1 - Yq0), torch.abs(Yd2 - Yd0), torch.abs(Yq2 - Yq0)))) / delta
      max_sens = torch.max (torch.stack ((max_sens, sens)))

      if bPrint: # and False:
        print (' * Yd=', Yd0, Yd1, Yd2)
        print ('   Yq=', Yq0, Yq1, Yq2)
        print ('   sens=', sens, 'max=', max_sens)
        print ('   vals', vals)

    sens_loss = max_sens * self.sensitivity['weight']
    #sens_loss = torch.max(max_sens - self.sensitivity['limit'], sens_floor)
    if bPrint:
      print ('  sens=', max_sens, 'loss=', sens_loss)
    return sens_loss, max_sens

  def trainModelCoefficients(self, bMAE = False):
    """Supervises the model fitting and validation using Adam optimizer.

    A *PVInvDataset* is created from *data_train*, which supports *DataLoader*s from *PyTorch* for training and validation.
    Batch sizes, continuation from a previous training run, and other options may be configured in the JSON file.
    Sensitivity and clamping may be added to the fitting loss, if configured in the JSON file.

    Args:
      bMAE (bool): if *True*, optimize the mean absolute error (MAE) instead of root mean square error (RMSE)

    Yields:

      - Block coefficients are trained in *pkl* files in the *model_folder*, and ready to export.
      - Training loss components written to 'Loss.npy' in the *model_folder*
      - After each epoch, if the current fitting loss is the best so far, model coefficients will be saved to *bestF1.pkl*, *bestF2.pkl*, and *bestH1.pkl*

    Returns:
      float: total training time elapsed between calls to *time.time()*
      list(float): fitting loss for each epoch
      list(float): validation loss for each epoch
      list(float): sensitivity loss for each epoch
    """
    if self.sensitivity is not None:
      self.setup_sensitivity_losses()

    if self.clamps is not None:
      self.setup_clamping_losses()

    self.optimizer = torch.optim.Adam([
      {'params': self.F1.parameters(), 'lr': self.lr},
      {'params': self.H1.parameters(), 'lr': self.lr},
      {'params': self.F2.parameters(), 'lr': self.lr},
    ], lr=self.lr, eps=self.eps)
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)

    # split the data into training and validation datasets
    total_ds = PVInvDataset (self.data_train, in_size, out_size, self.n_pad)
    nvalidation = int (len(total_ds) * float (self.n_validation_pct) / 100.0)
    ntraining = len(total_ds) - nvalidation
    splits = [ntraining, nvalidation]
    train_ds, valid_ds = torch.utils.data.random_split (total_ds, splits, 
                                                        generator=torch.Generator().manual_seed(self.n_validation_seed))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
    validation_scale = float(len(train_ds)) / float(len(valid_ds))
    sensitivity_scale = float(self.batch_size) / float(len(train_ds))
    print ('Dataset split:', len(total_ds), len(train_ds), len (valid_ds), 'validation_scale={:.3f}'.format(validation_scale))
    average_loss_divisor = 2.0 * float(ntraining) / float(self.batch_size)
    print ('Average loss scaling: ntrain={:d}, nbatch={:d}, average_loss_divisor={:.3f}'.format (ntraining, self.batch_size, average_loss_divisor))
    if self.sensitivity is not None:
      print ('Sensitivity scale={:.3f}'.format(sensitivity_scale))

    if self.continue_iterations:
      print ('continuing iterations on existing model coefficients')
      self.loadModelCoefficients()

    LOSS = []
    VALID = []
    SENS = []
    lossfile = os.path.join(self.model_folder, "Loss.npy")
    start_time = time.time()
    best_loss = 1.0e9
    for itr in range(0, self.num_iter):
      epoch_loss = 0.0
      epoch_sens = 0.0
      epoch_clamp = 0.0
      epoch_sigma = 0.0
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

        # add clamping loss to the fitting loss
        loss_clamp = torch.tensor (0.0)
        if self.clamps is not None:
          p1 = torch.maximum (self.clamping_zeros, y_hat - self.clamping_upper)
          p2 = torch.maximum (self.clamping_zeros, self.clamping_lower - y_hat)
          loss_clamp = self.t_step * torch.sum(p1 + p2) # [case, output]
#         loss_clamp = self.t_step * torch.sum(p1 + p2, dim=1) # [case, output]
#         total_loss = total_loss + torch.sum (loss, dim=0)
          loss_fit = loss_fit + loss_clamp

        # Compute the sensitivity loss
        loss_sens = torch.tensor (0.0)
        if self.sensitivity is not None:
          loss_sens, sigma = self.calc_sensitivity_losses (bPrint=False)
          if sigma > epoch_sigma:
            epoch_sigma = sigma

        # Optimize on this batch
        loss = loss_fit + loss_sens
#       print (' loss_fit', loss_fit)
#       print (' loss_sens', loss_sens)
#       print (' loss', loss)
        loss.backward(retain_graph=True) # for sensitivity optimization
        self.optimizer.step()
#        print ('  batch size={:d} loss={:12.6f}'.format (ub.shape[0], loss_fit))
        epoch_loss += loss_fit
        epoch_sens += loss_sens
        epoch_clamp += loss_clamp
#        print ('  fit,sens,loss = [{:12.6f} {:12.6f} {:12.6f}]'.format (loss_fit, loss_sens, loss))

#      print ('  last batch loss_fit, loss_sens, loss:', loss_fit, loss_sens, loss)
      # epoch_sens *= sensitivity_scale
      LOSS.append(epoch_loss.item())
      SENS.append(epoch_sens.item())

      # validation loss (sensitivity loss won't change, as it does not depend on the validation dataset)
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
        # add the clamping loss
        if self.clamps is not None:
          p1 = torch.maximum (self.clamping_zeros, y_hat - self.clamping_upper)
          p2 = torch.maximum (self.clamping_zeros, self.clamping_lower - y_hat)
          validation_clamp = self.t_step * torch.sum(p1 + p2) # [case, output]
          loss_fit = loss_fit + validation_clamp

        valid_loss += loss_fit * validation_scale

      VALID.append(valid_loss.item())
      total_loss = valid_loss.item() + epoch_loss.item()
      total_rmse = math.sqrt (total_loss / average_loss_divisor)
      total_loss += epoch_sens.item()
      if total_loss < best_loss:
        best_loss = total_loss
#        print (' == Saving the best model so far at iteration {:d}'.format (itr))
        self.saveModelCoefficients('best')
      if itr % self.print_freq == 0:
        print('Epoch {:4d} of {:4d} | TrLoss {:12.6f} | VldLoss {:12.6f} | SensLoss {:12.6f} | RMSE {:12.4f} | Sigma {:12.4f}'.format (itr, self.num_iter, epoch_loss, valid_loss, epoch_sens, total_rmse, epoch_sigma))
        self.saveModelCoefficients()
        np.save (lossfile, [LOSS, VALID, SENS])

    train_time = time.time() - start_time
    np.save (lossfile, [LOSS, VALID, SENS])
    return train_time, LOSS, VALID, SENS

  def saveModelCoefficients(self, prefix=''):
    """Save the current block coefficients

    Default file names are *F1.pkl*, *F2.pkl*, and *H1.pkl* in *model_folder*.

    Args:
      prefix(str): set as 'best' for saving the epoch with lowest fitting loss.
    """
    torch.save(self.F1.state_dict(), os.path.join(self.model_folder, prefix + "F1.pkl"))
    torch.save(self.H1.state_dict(), os.path.join(self.model_folder, prefix + "H1.pkl"))
    torch.save(self.F2.state_dict(), os.path.join(self.model_folder, prefix + "F2.pkl"))

  def loadModelCoefficients(self):
    """Load the block coefficients from *F1.pkl*, *F2.pkl* and *H1.pkl* in *model_folder*.
    """
    B1 = torch.load(os.path.join(self.model_folder, "F1.pkl"), weights_only=False)
    self.F1.load_state_dict(B1)
    B2 = torch.load(os.path.join(self.model_folder, "H1.pkl"), weights_only=False)
    self.H1.load_state_dict(B2)
#   print (self.H1)
#   print ('b_coeff', self.H1.b_coeff)
#   print ('rho', self.H1.rho)
#   print ('psi', self.H1.psi)
    B3 = torch.load(os.path.join(self.model_folder, "F2.pkl"), weights_only=False)
    self.F2.load_state_dict(B3)

  def make_H1Q1s(self, Hz):
    """Convert discrete-time H1(z) to continuous-time H1(s)

    Uses the Harold package to convert each transfer function
    between pairs of input and output channels. The transfer
    functions are undiscretized using the 'forward euler' method.
    Each continuous-time transfer function is checked for unstable poles,
    and if any are found, a warning message is printed. If such
    warning messages appear, the continuous-time model should not be used.
    One option is to re-train the blocks using the 'stable2ndx' *gtype*. 

    Args:
      Hz(MimoLinearDynamicalOperator): should be *H1*

    Returns:
      dict: H1(s) as rational transfer function coefficient arrays, keyed by 'a_i_j' for denominator and 'b_i_j' for numerator, where *i* is the *H1* output channel number and *j* is the *H1* input channel number 
      dict: Q1(s) as the state transition matrices for evaluating *x_dot = Ax + Bu; y = Cx + Du*, keyed by 'A_i_j', 'B_i_j', 'C_i_j', 'D_i_j', where *i* is the *H1* output channel number and *j* is the *H1* input channel number
    """
    if self.gtype == 'fir':
      return None, None
    n_in = Hz.in_channels
    n_out = Hz.out_channels
    n_a = Hz.n_a
    n_b = Hz.n_b
    H1s = {'n_in':n_in, 'n_out':n_out, 'n_a': n_a+1, 'n_b': n_b}
    Q1s = {'n_in':n_in, 'n_out':n_out, 'n_a': n_a, 'n_b': n_b}
    if self.gtype == 'iir':
      btf, atf = Hz.get_tfdata()
      b_coeff, a_coeff = Hz.__get_ba_coeff__()
#   print ('btf', btf.shape, btf[0][0])
#   print ('b_coeff', b_coeff.shape, b_coeff[0][0])
#   print ('atf', atf.shape, atf[0][0])
#   print ('a_coeff', a_coeff.shape, a_coeff[0][0])
      a = atf
      b = b_coeff
    elif self.gtype == 'stable2nd':
      block = Hz.state_dict()
      b = block['b_coeff'].numpy().squeeze()
      rho = block['rho'].numpy().squeeze()
      psi = block['psi'].numpy().squeeze()
      r = 1 / (1 + np.exp(-rho))
      beta = np.pi / (1 + np.exp(-psi))
      a1 = -2 * r * np.cos(beta)
      a2 = r * r
      a = np.ones ((b.shape[0], b.shape[1], 3)) 
      a[:,:,1] = a1
      a[:,:,2] = a2
    elif self.gtype == 'stable2ndx':
      block = Hz.state_dict()
      b = block['b_coeff'].numpy().squeeze()
      alpha1 = block['alpha1'].numpy().squeeze()
      alpha2 = block['alpha2'].numpy().squeeze()
      a1 = 2.0 * np.tanh(alpha1)
      a1abs = np.abs(a1)
      a2 = a1abs + (2.0 - a1abs) * 1.0/(1+np.exp(-alpha2)) - 1.0
      a = np.ones ((b.shape[0], b.shape[1], 3)) 
      a[:,:,1] = a1
      a[:,:,2] = a2

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
    """Writes the trained model coefficients to a JSON file.

    Includes all the training configuration parameters, and both continuous-time
    and discrete-time variants of *H1*.

    Args:
      filename (str): path and file name to the exported JSON file; typically ends in *_fhf.json*
    """
    config = {'name':'PV3', 'type':'F1+H1+F2', 't_step': self.t_step}
    config['normfacs'] = {}
    for key, val in self.normfacs.items():
      config['normfacs'][key] = {'scale':val['scale'], 'offset':val['offset'], 'max':val['max'], 'min':val['min']}
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
    config['clamps'] = self.clamps
    config['sensitivity'] = self.sensitivity

    self.append_lti (config, 'H1', self.H1)
    config['H1s'], config['Q1s'] = self.make_H1Q1s(self.H1)
    self.append_net (config, 'F1', self.F1)
    self.append_net (config, 'F2', self.F2)

    fp = open (filename, 'w')
    json.dump (config, fp, indent=2)
    fp.close()

  def printStateDicts(self):
    """Debugging output of the *F1*, *F2*, and *H1* block parameters in memory.
    """
    print ('F1', self.F1.state_dict())
    print ('F2', self.F2.state_dict())
    print ('H1', self.H1.in_channels, self.H1.out_channels, self.H1.n_a, self.H1.n_b, self.H1.n_k)
    print (self.H1.state_dict())

  def testOneCase(self, case_idx, npad, bUseTorchDS=False, bLog=True):
    """Compare the estimated and actual output for one of the cases used for training and validation.

    This function uses *PyTorch* to evaluate *H1*.

    See Also:
      :func:`stepOneCase`

    Args:
      case_idx(int): zero-based case number from *data_train* to test
      npad(int): number of decimated time points to pre-pad with initial condition values
      bUseTorchDS(bool): if *True*, use the torch tensor *DataLoader*, which matches the training process but takes longer and does not properly match the initial conditions. If *False*, perform test on *data_train* directly.
      bLog(bool): if *True*, output diagnostics on the dataset sizes

    Returns:
      float: root mean square error for this case
      float: mean absolute error for this case
      array_like(float,ndim=2): estimated output (y_hat) for this case, first dimension is time, second dimension is channel 
      array_like(float,ndim=2): true output (y) for this case, first dimension is time, second dimension is channel 
      array_like(float,ndim=2): input (u) for this case, first dimension is time, second dimension is channel 
    """
    if bUseTorchDS: # TODO: this doesn't match the correct Y[0] values
      total_ds = PVInvDataset (self.data_train, len(self.COL_U), len(self.COL_Y), npad, bLog=bLog)
      case_data = total_ds[case_idx]
      if bLog:
        print ('case_data shape', len(case_data))
        print ('case_data[0]', case_data[0].shape)
        print ('case_data[1]', case_data[1].shape)
      ub = torch.unsqueeze(case_data[0], dim=0)
      y_true_ret = case_data[1].detach().numpy()[npad:,:]
      y_true_err = case_data[1].detach().numpy()[self.n_loss_skip:,:]
      u_ret = case_data[0].detach().numpy()[npad:,:]
      if bLog:
        print ('Using PVInvDataset for IC padding:')
    else:
      case_data = self.data_train[[case_idx],:,:]
      udata = case_data[0,:,self.idx_in].squeeze().transpose()
      # padding initial conditions
      ic = np.zeros((npad, len(self.idx_in)), dtype=case_data.dtype)
      for i in range(len(self.idx_in)):
        ic[:,i] = udata[0,i]
      udata = np.concatenate ((ic, udata))
      ub = torch.tensor (np.expand_dims (udata, axis=0), dtype=torch.float)
      u_ret = udata[npad:,:]
      y_true_ret = np.transpose(case_data[0,:,self.idx_out]).squeeze()
      # padding initial conditions on the output vector for error metrics
      ic = np.zeros((npad - self.n_loss_skip, len(self.idx_out)), dtype=case_data.dtype)
      for i in range(len(self.idx_out)):
        ic[:,i] = y_true_ret[0,i]
      y_true_err = np.concatenate ((ic, y_true_ret))
      if bLog:
        print ('Using data_train for IC padding:')
    if bLog:
      print ('  ub.shape', ub.shape)
      print ('  y_true.shape', y_true_ret.shape)
      print ('  y_err.shape', y_true_err.shape)
      print ('  u_ret.shape', u_ret.shape)

    # model evaluation on the padded data
    y_non = self.F1 (ub)
    y_lin = self.make_mimo_ylin (y_non)
    y_hat = self.F2 (y_lin)

    # error metrics
    y_hat_ret = y_hat.detach().numpy()[[0], npad:, :].squeeze()
    y_hat_err = y_hat.detach().numpy()[[0], self.n_loss_skip:, :].squeeze()
    rmse = dynonet.metrics.error_rmse(y_true_err, y_hat_err)
    mae = dynonet.metrics.error_mae(y_true_err, y_hat_err)
    return rmse, mae, y_hat_ret, y_true_ret, u_ret

  def simulateVectors(self, T, G, Fc, Md, Mq, Vrms, GVrms, Ctl, npad):
    """Forward evaluation by processing individual vectors of input.

    *Deprecated*: this function was only used with early Norton models that used
    *Vrms* instead of *Vd* and *Vq* inputs.

    Args:
      T(list(float)): array of input temperature vs. time
      G(list(float)): array of input solar irradiance vs. time
      Fc(list(float)): array of input frequency vs. time
      Md(list(float)): array of input d-axis voltage control index vs. time
      Mq(list(float)): array of input q-axis voltage control index vs. time
      Vrms(list(float)): array of input RMS voltage vs. time
      GVrms(list(float)): array of input polynomial feature vs. time
      Ctl(list(float)): array of input control model vs. time
      npad(int): number of decimated-in-time points to pre-pad with initial values

    Return:
      list(float): DC voltage vs. time
      list(float): DC current vs. time
      list(float): Id vs. time
      list(float): Iq vs. time
    """
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

    #REVISIT: hard-wired with 4 outputs in the order shown, but note this function has been deprecated.
    # To fix, see the approach in the local steady_state_response function.
    Vdc = self.de_normalize (y_hat[npad:,0], self.normfacs['Vdc'])
    Idc = self.de_normalize (y_hat[npad:,1], self.normfacs['Idc'])
    ACd = self.de_normalize (y_hat[npad:,2], self.normfacs[self.d_key])
    ACq = self.de_normalize (y_hat[npad:,3], self.normfacs[self.q_key])
    return Vdc, Idc, ACd, ACq

  def stepOneCase(self, case_idx):
    """Compare the estimated and actual output for one of the cases used for training and validation.

    This function uses the IIR filter coefficients and history terms to evaluate *H1*.
    The history terms allow proper initialization of *H1* without pre-padding the data
    with initial values.

    See Also:
      :func:`testOneCase`

    Args:
      case_idx(int): zero-based case number from *data_train* to test

    Returns:
      float: root mean square error for this case
      float: mean absolute error for this case
      array_like(float,ndim=2): estimated output (y_hat) for this case, first dimension is time, second dimension is channel 
      array_like(float,ndim=2): true output (y) for this case, first dimension is time, second dimension is channel 
      array_like(float,ndim=2): input (u) for this case, first dimension is time, second dimension is channel 
    """
    case_data = self.data_train[case_idx,:,:]
    n = len(self.t)
    y_hat = np.zeros(shape=(n,len(self.idx_out)), dtype=case_data.dtype)
    ub = torch.zeros((1, 1, len(self.idx_in)), dtype=torch.float)

    self.start_simulation()
    # establish initial conditions of the normalized data
    ub = torch.tensor (case_data[0,self.idx_in]) # initial inputs
    with torch.no_grad():
      y_non = self.F1 (ub)
      for i in range(self.H1.out_channels):
        for j in range(self.H1.in_channels):
          ynew = y_non[j] * np.sum(self.b_coeff[i,j,:]) / (np.sum(self.a_coeff[i,j,:])+1.0)
          self.uhist[i][j][:] = y_non[j]
          self.yhist[i][j][:] = ynew

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
    y_true = case_data[:,self.idx_out].squeeze()
    udata = case_data[:,self.idx_in].squeeze()
    rmse = dynonet.metrics.error_rmse(y_true, y_hat)
    mae = dynonet.metrics.error_mae(y_true, y_hat)

    return rmse, mae, y_hat, y_true, udata

  def setup_clamping_losses(self):
    """Normalize the output channel clamping limits and identify the clamped channel numbers, in preparation for adding clamping losses to fitting losses in model training.
    """
    sizes = (self.batch_size, self.data_train.shape[1], len(self.COL_Y))
    self.clamping_zeros = torch.zeros(sizes, requires_grad=True)
    lower_np = np.zeros(sizes)
    upper_np = np.zeros(sizes)
    for i in range(len(self.COL_Y)):
      key = self.COL_Y[i]
      val = self.clamps[key]
      lower_np[:,:,i] = self.normalize (val[0], self.normfacs[key])
      upper_np[:,:,i] = self.normalize (val[1], self.normfacs[key])
    self.clamping_lower = torch.from_numpy (lower_np)
    self.clamping_upper = torch.from_numpy (upper_np)

  def clamping_losses(self):
    """ Calculate the clamping losses with torch functions that support optimization.

    *Deprecated*. If any output channel departs from its clamped range, the clamping
    losses are non-zero. However, this creates step changes that lead to oscillations
    in the total loss. The normal fitting losses, together with improvements in the
    initialization of *H1*, do a better job of constraining outputs to the expected range.

    Returns:
      list(float): total clamping loss over all cases and times, by output channel
    """
    total_loss = torch.zeros (out_size, requires_grad=True)

    for ub, y_true in total_dl: # batch loop
      y_non = self.F1 (ub)
      y_lin = self.make_mimo_ylin (y_non)
      y_hat = self.F2 (y_lin)
      p1 = torch.maximum (zeros, y_hat - upper)
      p2 = torch.maximum (zeros, lower - y_hat)
      loss = self.t_step * torch.sum(p1 + p2, dim=1) # [case, output]
      total_loss = total_loss + torch.sum (loss, dim=0)
#     print (total_loss)

    return total_loss

  def clampingErrors(self, bByCase=False):
    """Summarize the clamping errors in a trained model

    Args:
      bByCase(bool): *True* if the individual clamping loss per case is desired, otherwise just the total is calculated.

    Returns:
      list(float): total clamping loss over all cases and times, by output channel
      array_like(float,ndim=2): case clamping loss over all times. First dimension is case number, second dimension is output channel. *None* if *bByCase=False*
    """
    self.n_cases = self.data_train.shape[0]
    npts = self.data_train.shape[1]
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)
    sizes = (self.batch_size, npts, out_size)

    # normalize the limits
    zeros = torch.zeros(sizes, requires_grad=True)
    lower_np = np.zeros(sizes)
    upper_np = np.zeros(sizes)
    for i in range(out_size):
      key = self.COL_Y[i]
      val = self.clamps[key]
      lower_np[:,:,i] = self.normalize (val[0], self.normfacs[key])
      upper_np[:,:,i] = self.normalize (val[1], self.normfacs[key])
    lower = torch.from_numpy (lower_np)
    upper = torch.from_numpy (upper_np)
#   print ('Clamping shapes for zeros, lower, upper', zeros.shape, lower.shape, upper.shape)

    total_loss = torch.zeros (out_size, requires_grad=True)
    if bByCase:
      case_loss = np.zeros([self.n_cases, out_size])
    else:
      case_loss = None

    total_ds = PVInvDataset (self.data_train, in_size, out_size, self.n_pad, bLog=False)
    total_dl = torch.utils.data.DataLoader(total_ds, batch_size=self.batch_size, shuffle=False)

    icase = 0
    for ub, y_true in total_dl: # batch loop
      y_non = self.F1 (ub)
      y_lin = self.make_mimo_ylin (y_non)
      y_hat = self.F2 (y_lin)
      p1 = torch.maximum (zeros, y_hat - upper)
      p2 = torch.maximum (zeros, lower - y_hat)
      loss = self.t_step * torch.sum(p1 + p2, dim=1) # [case, output]
      total_loss = total_loss + torch.sum (loss, dim=0)
#     print (total_loss)

      # extract the clamping losses by case, but only for numpy reporting
      if bByCase:
        cl = loss.detach().numpy()
        nb = cl.shape[0]
        iend = icase + nb
        case_loss[icase:iend,:] = cl[:,:]
        icase = iend
    return total_loss, case_loss

  def trainingErrors(self, bByCase=False):
    """Summarize the **fitting** errors in a trained model.

    Evaluates both root mean square error (RMSE) and mean absolute error (MAE).

    Args:
      bByCase(bool): *True* if the individual fitting loss per case is desired, otherwise just the total is calculated.

    Returns:
      list(float): total fitting RMSE over all cases and times, by output channel
      list(float): total fitting MAE over all cases and times, by output channel
      array_like(float,ndim=2): case RMSE over all times. First dimension is case number, second dimension is output channel. *None* if *bByCase=False*
      array_like(float,ndim=2): case MAE over all times. First dimension is case number, second dimension is output channel. *None* if *bByCase=False*
    """
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

    total_ds = PVInvDataset (self.data_train, in_size, out_size, self.n_pad, bLog=False)
    total_dl = torch.utils.data.DataLoader(total_ds, batch_size=self.batch_size, shuffle=False)

    icase = 0
    for ub, y_true in total_dl: # batch loop
      y_non = self.F1 (ub)
      y_lin = self.make_mimo_ylin (y_non)
      y_hat = self.F2 (y_lin)
      y1 = y_true.detach().numpy()[:,self.n_loss_skip:,:]
      y2 = y_hat.detach().numpy()[:,self.n_loss_skip:,:]
      y_err = np.abs(y1-y2)
      y_sqr = y_err*y_err
      nb = y_err.shape[0]
      npts = y_err.shape[1]
      ncol = y_err.shape[2]
#     print ('** Training Error Batch', nb, npts, ncol, y1.shape, y2.shape)
      mae = np.mean (y_err, axis=1) # nb x ncol
      mse = np.mean (y_sqr, axis=1)
      total_mae += np.sum(mae, axis=0)
      total_rmse += np.sum(mse, axis=0)
#     print ('   Accumulating case squared errors', mse.shape, total_rmse)
      if bByCase:
        iend = icase + nb
        case_mae[icase:iend,:] = mae[:,:]
        case_rmse[icase:iend,:] = mse[:,:]
#       print ('    Slicing case squared errors', icase, iend, case_rmse.shape)
        icase = iend
    total_rmse = np.sqrt(total_rmse / self.n_cases)
#   print ('** Summarizing RMSE', total_rmse, self.n_cases, total_rmse)
    total_mae /= self.n_cases
    if bByCase:
      case_rmse = np.sqrt(case_rmse)
#   print ('==========Detailing the first case results for Idc==========')
#   print ('k,Idc_true,Idc_hat,y_err,y_sqr')
#   for k in range(npts):
#     print ('{:4d},{:.6f},{:.6f},{:.6f},{:.6f}'.format (k, y1[0,k,1], y2[0,k,1], y_err[0,k,1], y_sqr[0,k,1]))
    return total_rmse, total_mae, case_rmse, case_mae

  def trainingLosses(self):
    """Evaluates the **fitting** loss of a trained model.

    Returns:
      float: total RMSE over all cases, times, and output channels
    """
    self.n_cases = self.data_train.shape[0]
    in_size = len(self.COL_U)
    out_size = len(self.COL_Y)

    total_ds = PVInvDataset (self.data_train, in_size, out_size, self.n_pad, bLog=False)
    total_dl = torch.utils.data.DataLoader(total_ds, batch_size=self.batch_size, shuffle=False)
    rmse_loss = 0.0

    for ub, y_true in total_dl: # batch loop
      y_non = self.F1 (ub)
      y_lin = self.make_mimo_ylin (y_non)
      y_hat = self.F2 (y_lin)
      if self.n_loss_skip > 0:
        err_fit = y_true[:,self.n_loss_skip:,:] - y_hat[:,self.n_loss_skip:,:]
      else:
        err_fit = y_true - y_hat
      loss_fit = torch.mean(err_fit**2)
      rmse_loss += loss_fit

    return rmse_loss.item()

  def set_LCL_filter(self, Lf, Cf, Lc):
    """Set the inverter output filter parameters

    These may be used to back-calculate voltages and currents at the voltage-source
    converter (VSC) terminals, based on voltages and currents at the AC terminals, i.e.,
    at the point of commong coupling. However, this is not a causal operation.

    Args:
      Lf(float): filter inductance between the VSC and filter capacitor
      Cf(float): filter capacitance
      Lc(float): filter inductance between the filter capacitor and the PCC
    """
    self.Lf = Lf
    self.Cf = Cf
    self.Lc = Lc

  def check_poles(self):
    """Verify that the H(z) transfer function poles are stable.

    All poles must lie within the unit circle in the Z plane. This does not
    guarantee that the poles of H1(s) will be stable. Uses the *control* package
    *TransferFunction* class to check the poles.

    See Also:
      :func:`make_H1Q1s`

    Yields:
      Printed output will contain the word 'UNSTABLE' if any unstable poles were found.
    """
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
    elif self.gtype == 'stable2ndx':
      b = self.H1.b_coeff.detach().numpy().squeeze()
      n_in = b.shape[1]
      n_out = b.shape[0]
      alpha1 = self.H1.alpha1.detach().numpy().squeeze()
      alpha2 = self.H1.alpha2.detach().numpy().squeeze()
      a1 = 2.0 * np.tanh(alpha1)
      a1abs = np.abs(a1)
      a2 = a1abs + (2.0 - a1abs) * 1.0/(1+np.exp(-alpha2)) - 1.0
      a = np.ones ((b.shape[0], b.shape[1], 3))
      a[:,:,1] = a1
      a[:,:,2] = a2
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
    """Sets up the *H1* history terms for an efficient IIR simulation of the trained model

    Args:
      bPrint(bool): if *True*, print some information about the *H1* IIR setup

    Returns:
      list(str): names of input columns to help the user construct input vectors at each discrete time step
    """
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
      print ('  start_simulation [n_a, n_b, n_in, n_out]=[{:d} {:d} {:d} {:d}]'.format (self.H1.n_a, self.H1.n_b, self.H1.in_channels, self.H1.out_channels))
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
        print ('    constructed a_coeff')                                         
    elif self.gtype == 'stable2ndx' and not hasattr(self.H1, 'a_coeff'):
      alpha1 = self.H1.alpha1.detach().numpy().squeeze()
      alpha2 = self.H1.alpha2.detach().numpy().squeeze()
      a1 = 2.0 * np.tanh(alpha1)
      a1abs = np.abs(a1)
      a2 = a1abs + (2.0 - a1abs) * 1.0/(1+np.exp(-alpha2)) - 1.0
      self.a_coeff = np.ones ((self.b_coeff.shape[0], self.b_coeff.shape[1], 2)) #  3))
      self.a_coeff[:,:,0] = a1
      self.a_coeff[:,:,1] = a2
      if bPrint:
        print ('    constructed a_coeff')                                         
    else:
      self.a_coeff = self.H1.a_coeff.detach().numpy()
      if bPrint:
        print ('    existing a_coeff')
    if bPrint:
      print ('    a=\n', self.a_coeff)
      print ('    b=\n', self.b_coeff)
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
                                             
# set up the static nonlinearity blocks for time step simulation (TODO) why after return value?
    self.F1.eval()
    self.F2.eval()

  def normalize (self, val, fac):
    """Normalizes one channel for the trained blocks.

    Args:
      list(float): un-normalized channel data
      fac(dict): a member of *normfacs* with *scale* and *offset*

    Returns:
      list(float): normalized channel data
    """
    return (val - fac['offset']) / fac['scale']

  def de_normalize (self, val, fac):
    """De-normalizes one channel from the trained blocks to the user.

    Args:
      list(float):normalized channel data
      fac(dict): a member of *normfacs* with *scale* and *offset*

    Returns:
      list(float): de-normalized channel data
    """
    return val * fac['scale'] + fac['offset']

  def steady_state_response (self, vals):
    """Calculate steady-state block outputs using the IIR coefficients of *H1*.

    Args:
      vals(list(float)): vector of de-normalized inputs, in order to match *COL_U*

    Returns:
      float: steady-state DC voltage (if present in the output)
      float: steady-state DC current (if present in the output)
      float: steady-state *Id* for a Norton model or *Vd* for a Thevenin model
      float: steady-state *Iq* for a Norton model or *Vq* for a Thevenin model
    """
    for i in range(len(vals)):
      vals[i] = self.normalize (vals[i], self.normfacs[self.COL_U[i]])

    print (self.d_key, self.q_key)

    ub = torch.tensor (vals, dtype=torch.float)
    with torch.no_grad():
      y_non = self.F1 (ub)
      self.ysum[:] = 0.0
      for i in range(self.H1.out_channels):
        for j in range(self.H1.in_channels):
          ynew = y_non[j] * np.sum(self.b_coeff[i,j,:]) / (np.sum(self.a_coeff[i,j,:])+1.0)
          self.ysum[i] += ynew
      y_lin = torch.tensor (self.ysum, dtype=torch.float)
      y_hat = self.F2 (y_lin)

    if len(y_hat) < 3:
      ACd = y_hat[1].item()
      ACq = y_hat[2].item()
      ACd = self.de_normalize (ACd, self.normfacs[self.d_key])
      ACq = self.de_normalize (ACq, self.normfacs[self.q_key])
      return ACd, ACq

    if len(y_hat) < 4:
      Idc = y_hat[0].item()
      ACd = y_hat[1].item()
      ACq = y_hat[2].item()
      Idc = self.de_normalize (Idc, self.normfacs['Idc'])
      ACd = self.de_normalize (ACd, self.normfacs[self.d_key])
      ACq = self.de_normalize (ACq, self.normfacs[self.q_key])
      return Idc, ACd, ACq

    Vdc = y_hat[0].item()
    Idc = y_hat[1].item()
    ACd = y_hat[2].item()
    ACq = y_hat[3].item()

    Vdc = self.de_normalize (Vdc, self.normfacs['Vdc'])
    Idc = self.de_normalize (Idc, self.normfacs['Idc'])
    ACd = self.de_normalize (ACd, self.normfacs[self.d_key])
    ACq = self.de_normalize (ACq, self.normfacs[self.q_key])

    return Vdc, Idc, ACd, ACq

  def step_simulation (self, vals, nsteps=1):
    """Simulate one or more discrete time steps of a trained model using the IIR coefficients of *H1*

    Args:
      vals(list(float)): vector of de-normalized inputs, in order to match *COL_U*
      nsteps(int): number of discrete time steps to simulate. This can be > 1 for brute-force initialization, but afterward it is usually 1.

    Returns:
      float: DC voltage
      float: DC current
      float: *Id* for a Norton model or *Vd* for a Thevenin model
      float: *Iq* for a Norton model or *Vq* for a Thevenin model
    """
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
      ACd = y_hat[1].item()
      ACq = y_hat[2].item()
      Idc = self.de_normalize (Idc, self.normfacs['Idc'])
      ACd = self.de_normalize (ACd, self.normfacs[self.d_key])
      ACq = self.de_normalize (ACq, self.normfacs[self.q_key])
      return Idc, ACd, ACq

    Vdc = y_hat[0].item()
    Idc = y_hat[1].item()
    ACd = y_hat[2].item()
    ACq = y_hat[3].item()

    Vdc = self.de_normalize (Vdc, self.normfacs['Vdc'])
    Idc = self.de_normalize (Idc, self.normfacs['Idc'])
    ACd = self.de_normalize (ACd, self.normfacs[self.d_key])
    ACq = self.de_normalize (ACq, self.normfacs[self.q_key])

#   if self.Lf is not None:
#     Ic = np.complex (Irms+0.0j)
#     Vf = Vc + ZLc * Ic
#     If = Vf / ZCf
#     Is = Ic + If
#     Vs = Vf + ZLf * Is
#   else:
#     Vs = Vc
#     Is = np.complex (Irms+0.0j)

    return Vdc, Idc, ACd, ACq

