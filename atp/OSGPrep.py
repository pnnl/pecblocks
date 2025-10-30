# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: OSGPrep.py
""" Creates polynomial features and control mode signal for training single-phase OSG model.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import h5py
import math

# for decimation
#ndec=400
#b, a = signal.butter (2, 1.0 / 4096.0, btype='lowpass', analog=False)
filtered = ['Vrms', 'Irms', 'Id', 'Iq', 'Vd', 'Vq', 'Vdc', 'Idc']
filtered = [] # signals already filtered

ctl_t = [0.00, 3.99, 4.00, 1000.00]
ctl_y = [0.00, 0.00, 1.00, 1.00]

def load_or_zero_signal (grp, key, dlen):
  y = np.zeros(dlen)
  if key in grp:
    grp[key].read_direct (y)
  if key in filtered:
    return signal.lfilter (b, a, y)[::ndec]
#  return y[::ndec]
  return y

def impute_control_mode (t):
  ctl = np.zeros(len(t))
  for i in range(len(t)):
    ctl[i] = np.interp (t[i], ctl_t, ctl_y)
  return ctl

inputfiles = ['osg4.hdf5']
fout = h5py.File ('osg4_vdvq.hdf5', 'w')
idx = 0
for filename in inputfiles:
  fp = h5py.File(filename, 'r')
  for grp_name, grp in fp.items():
    print ('processing', grp_name)
    grpout = fout.create_group ('case{:d}'.format(idx))
    idx += 1
    sigs = {}
    dlen = grp['t'].len()
    traw = np.zeros(dlen)
    grp['t'].read_direct(traw)
#    sigs['t'] = traw[::ndec]
    sigs['t'] = traw
    for key in ['T', 'G', 'Fc', 'Ud', 'Uq', 'Vdc', 'Idc', 'Id', 'Iq', 'Vd', 'Vq', 'Vrms']:
      sigs[key] = load_or_zero_signal (grp, key, dlen)
    # create extra input features
    sigs['GVrms'] = 0.001 * sigs['G'] * sigs['Vrms']
    sigs['Ctl'] = impute_control_mode (sigs['t'])
    for key in ['t', 'T', 'G', 'Fc', 'Ud', 'Uq', 'Ctl', 'Vd', 'Vq', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq']:
      grpout.create_dataset (key, data=sigs[key], compression='gzip')

print ('created {:d} training cases'.format(idx))

