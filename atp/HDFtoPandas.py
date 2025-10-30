# Copyright (C) 2018-2021 Battelle Memorial Institute
# file: HDF5TrainingPlot.py
""" Plots the ATP training simulations from HDF5 files.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import numpy as np
import h5py
import pandas as pd

plot_defs = [
    {'row':0, 'col':0, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
    {'row':1, 'col':0, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
    {'row':2, 'col':0, 'tag':'P', 'title':'Real Power',  'ylabel':'kW', 'scale':0.001},
    {'row':0, 'col':1, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A'},
    {'row':2, 'col':1, 'tag':'Q', 'title':'Reactive Power',  'ylabel':'kVAR', 'scale':0.001},
    {'row':0, 'col':2, 'tag':'F0', 'title':'Nominal Frequency',  'ylabel':'Hz'},
    {'row':1, 'col':2, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
    {'row':2, 'col':2, 'tag':'F', 'title':'Measured Frequency',  'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'VmagA', 'title':'Va', 'ylabel':'Vrms'},
    {'row':1, 'col':3, 'tag':'VmagB', 'title':'Vb', 'ylabel':'Vrms'},
    {'row':2, 'col':3, 'tag':'VmagC', 'title':'Vc', 'ylabel':'Vrms'},
    {'row':0, 'col':4, 'tag':'ImagA', 'title':'Ia', 'ylabel':'Arms'},
    {'row':1, 'col':4, 'tag':'ImagB', 'title':'Ib', 'ylabel':'Arms'},
    {'row':2, 'col':4, 'tag':'ImagC', 'title':'Ic', 'ylabel':'Arms'},
    {'row':0, 'col':5, 'tag':'VangA', 'title':'VangA', 'ylabel':'rad'},
    {'row':1, 'col':5, 'tag':'VangB', 'title':'VangB', 'ylabel':'rad'},
    {'row':2, 'col':5, 'tag':'VangC', 'title':'VangC', 'ylabel':'rad'}
  ]

def add_group(df, grp):
  dlen = grp['t'].len()
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  df = df.append (pd.Series(t))
  for plot in plot_defs:
    grp[plot['tag']].read_direct (y)
    df = df.append (pd.Series(y))

filename = 'new.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

df = pd.DataFrame()

with h5py.File(filename, 'r') as f:
  for grp_name, grp in f.items():
    add_group (df, grp_name, grp)

print (df.describe())

