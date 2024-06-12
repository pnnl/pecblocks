# copyright 2021-2024 Battelle Memorial Institute
# HW model training and simulation code for 3-phase inverters
"""
  File support functions.
"""
import numpy as np
import pandas as pd
import os
import zipfile
import h5py
import math

def read_hdf5_file(filename, cols, n_dec=1, n_skip=0, n_trunc=0, prefix=None):
  """Adds each group to a list of Pandas dataframes.

  The HDF5 file should include 0..n groups, each with the same column
  keys, record length, and sample interval. One column key should be 't'.

  Args:
    filename (str): name of HDF5 file, one group per event record, groups indexed from 0
    cols (list(str)): list of column keys to extract from each group
    n_dec (int): decimation, i.e., take every n_dec point
    n_skip (int): number of samples, after decimation, to exclude from beginning of each event
    n_trunc (int): number of samples, after decimation, to exclude from end of each event
    prefix (str): optional prefix to the group number

  Returns:
    list(DataFrame): List of Pandas DataFrames, one per group.
  """
  pdata=[]
  with h5py.File(filename, 'r') as f:
    ngroups = len(f.items())
    for i in range(ngroups):
      if prefix is not None:
        key = '{:s}{:d}'.format(prefix,i)
      else:
        key = str(i)
      grp = f[key]
      vals = []
      ncols = len (cols)
      nrows = grp[cols[0]].len()
      ndfrows = int(math.ceil(nrows/n_dec))
      ary = np.zeros (shape=(ndfrows, ncols))
      j = 0
      for col in cols:
        x = np.zeros(nrows)
        grp[col].read_direct (x)
        ary[:,j] = x[::n_dec]
        j += 1
      df = pd.DataFrame (data=ary[n_skip:-n_trunc or None,:], columns=cols)
      pdata.append(df)
  return pdata

def read_csv_files(path, pattern=".csv"):
  """Adds a set of CSV files to a list of Pandas dataframes.

  Each CSV file should use the comma as separator, column names in the first row.
  The Pandas read_csv function is called on each CSV file.

  Args:
    path (str): this can be the name of a zip file, or a path to glob for files with *pattern* in the name.
    pattern (list(str)): the file extension to look for. Ignored for zip file input, and not a regular expression.

  Returns:
    list(DataFrame): List of Pandas DataFrames, one per CSV file.
  """
  if zipfile.is_zipfile (path):
    zf = zipfile.ZipFile (path)
    pdata =[]
    for zn in zf.namelist():
      pdata0 = pd.read_csv (zf.open(zn),sep=',',header=0,on_bad_lines='skip')
      if pdata0.shape[0] >0:
        pdata += [pdata0.copy()]
    return pd.concat(pdata)
  else:
    files = [fn for fn in os.listdir(path) if pattern in fn]; 
    # files = np.sort(files)
    if len(files)>0:
      pdata =[]
      for i in range(len(files)):
        pdata0 = pd.read_csv(os.path.join(path,files[i]),sep=',',header=0,on_bad_lines='skip')
        if pdata0.shape[0] >0:
          pdata += [pdata0.copy()]  
      return pd.concat(pdata)


