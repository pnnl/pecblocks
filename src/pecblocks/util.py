# copyright 2021-2024 Battelle Memorial Institute
# HW model training and simulation code for 3-phase inverters

import numpy as np
import pandas as pd
import os
import zipfile
import h5py
import math

'''adds each group to a list of Pandas dataframes'''
def read_hdf5_file(filename, cols, n_dec=1, n_skip=0, n_trunc=0, prefix=None):
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


