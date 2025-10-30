# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: BalancedPrep.py
""" Creates polynomial features and control mode signal for training balanced model.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math

ctl_t = [0.00, 2.00, 2.01, 1000.00]
ctl_y = [0.00, 0.00, 1.00, 1.00]

def load_or_zero_signal (grp, key, dlen):
  y = np.zeros(dlen)
  if key in grp:
    grp[key].read_direct (y)
  return y

def impute_control_mode (t):
  ctl = np.zeros(len(t))
  for i in range(len(t)):
    ctl[i] = np.interp (t[i], ctl_t, ctl_y)
  return ctl

inputfiles = ['data/big3raw.hdf5']
fout = h5py.File ('c:/data/big3.hdf5', 'w')
idx = 0
for filename in inputfiles:
  fp = h5py.File(filename, 'r')
  for grp_name, grp in fp.items():
    print ('processing', grp_name)
    grpout = fout.create_group ('case{:d}'.format(idx))
    idx += 1
    sigs = {}
    dlen = grp['t'].len()
    for key in ['t', 'T', 'G', 'Fc', 'Md', 'Mq', 'Vdc', 'Idc', 'Id', 'Iq', 'Vd', 'Vq']:
      sigs[key] = load_or_zero_signal (grp, key, dlen)
    # create extra input features
    sigs['Vrms'] = math.sqrt(1.5) * np.sqrt(sigs['Vd']*sigs['Vd'] + sigs['Vq']*sigs['Vq'])
    sigs['GVrms'] = 0.001 * sigs['G'] * sigs['Vrms']
    sigs['Ctl'] = impute_control_mode (sigs['t'])
    for key in ['t', 'T', 'G', 'Fc', 'Md', 'Mq', 'Ctl', 'Vd', 'Vq', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq']:
      grpout.create_dataset (key, data=sigs[key], compression='gzip')

print ('created {:d} training cases'.format(idx))

