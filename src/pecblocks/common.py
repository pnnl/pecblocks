# copyright 2021-2024 Battelle Memorial Institute
"""PyTorch dataloader customized to fitting generalized block diagram models. Implements **PVInvDataset** class.
"""
import torch
import os
import pandas as pd
import numpy as np

class PVInvDataset(torch.utils.data.Dataset):
  """Helper class to load training datasets for PyTorch.

  A list of Pandas dataframes is converted to a torch tensor for
  training the HWPV models. The tensor can be pre-padded with initial
  conditions before training the model.

  Attributes:
    data (tensor): data [case index, time point, channel index]
    len (int): number of time points per channel
    n_in (int): number of input channels
    n_out (int): number of output channels
  """

  def __init__(self, data, input_dim, out_dim, pre_pad = 0):
    """ Constructor method

    Total number of channels must be *input_dim + output_dim*.

    Args:
      data (list(DataFrame)): list of Pandas DataFrames from the HDF5 or CSV source (see *util* module).
      input_dim (int): number of input channels
      output_dim (int): number of output channels
      pre_pad (int): number of initial condition points to pre-pend, suggest 10-20% of the total number of time points.
    """
    if pre_pad > 0:
      # padding initial conditions
      nchan = input_dim + out_dim
      ic = np.zeros((data.shape[0], pre_pad, nchan), dtype=data.dtype)
      print ('prepending initial conditions', ic.shape, data.shape, ic.dtype, data.dtype)
      for i in range(nchan):
        ic[:,i] = data[0,i]
      icdata = np.concatenate ((ic, data), axis=1)
      print ('results in', icdata.shape)      
      self.data = torch.tensor(icdata)
    else:
      self.data = torch.tensor(data)
    self.len = self.data.shape[0]
    self.n_in = input_dim
    self.n_out = out_dim
 
  def __len__(self):
    """Retrieve number of cases or events

    Returns:
      int: number of cases (first dimension) in *data*
    """
    return self.len

  def __getitem__(self, idx):
    """Retrieve data for a case or event

    Args:
      idx (int): zero-based case number

    Returns:
      tensor: case data [number of time points, number of input channels plus output channels]
    """
    return self.data[idx, :, range(self.n_in)], self.data[idx, :, range(self.n_in,self.n_in+self.n_out)]

def read_csv_files_to_dflist(path,  pattern=".csv", time_range =None):
  """Helper function to load CSV files into a list of Pandas DataFrames.

  Each CSV file should use the comma as separator, column names in the first row.
  The Pandas read_csv function is called on each CSV file.

  Args:
    path (str): this must be a path to glob for files with *pattern* in the name.
    pattern (list(str)): the file extension to look for. Not a regular expression.

  Returns:
    list(DataFrame): List of Pandas DataFrames, one per CSV file.
  """
  pdata =[]
  files = [fn for fn in os.listdir(path) if pattern in fn]; 
  # files = np.sort(files)
  if len(files)>0:       
    for i in range(len(files)):
      pdata0 = pd.read_csv(os.path.join(path,files[i]),sep=',',header=0,error_bad_lines=False)
      if(time_range!=None):
        pdata0= pdata0[(pdata0['TIME']>time_range[0]) & (pdata0['TIME']<time_range[1])]
      if pdata0.shape[0] >0:
        pdata.append(pdata0) 
    return pdata

