# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: MergeUnbalanced.py
""" Processes time-varying dq0 signals into steady components for training unbalanced model.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import hilbert
import h5py
import math

ctl_t = [0.00, 2.00, 2.01, 1000.00]
ctl_y = [0.00, 0.00, 1.00, 1.00]

unb_t = [0.00, 3.00, 3.01, 1000.00]
unb_y = [0.00, 0.00, 1.00, 1.00]

def moving_average(a, n=10):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret / n
#  return ret[n - 1:] / n

#flt_b, flt_a = signal.butter (5, 0.05)
#flt_b, flt_a = signal.cheby1 (8, 5, 0.1)
#def my_filter(a):
#  return signal.filtfilt (a)

bord=6
blev=0.05
flt_sos_lp = signal.butter (bord, blev, btype='lowpass', output='sos')
flt_sos_hp = signal.butter (bord, blev, btype='highpass', output='sos')

def my_filter(a, hp=False):
  if hp:
    raw = np.abs(signal.sosfiltfilt (flt_sos_hp, a, padtype=None))
    avg = moving_average(raw, n=30)
    return signal.sosfiltfilt (flt_sos_lp, avg, padtype=None)
#    analytic_signal = hilbert(raw)
#    return np.abs(analytic_signal)
  else:
    return signal.sosfiltfilt (flt_sos_lp, a, padtype=None)

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

def impute_unbalanced_mode (t):
  unb = np.zeros(len(t))
  for i in range(len(t)):
    unb[i] = np.interp (t[i], unb_t, unb_y)
  return unb

inputfiles = ['balanced.hdf5', 'looped_hw8_1ph.hdf5']
fout = h5py.File ('unbalanced.hdf5', 'w')
idx = 0
for filename in inputfiles:
  fp = h5py.File(filename, 'r')
  for grp_name, grp in fp.items():
    grpout = fout.create_group ('case{:d}'.format(idx))
    idx += 1
    sigs = {}
    dlen = grp['t'].len()
    for key in ['t', 'T', 'G', 'Fc', 'Md', 'Mq', 'Vdc', 'Idc', 'Id', 'Iq', 'I0', 'Vd', 'Vq', 'V0']:
      sigs[key] = load_or_zero_signal (grp, key, dlen)
    # create extra input features
    sigs['Vrms'] = math.sqrt(1.5) * np.sqrt(sigs['Vd']*sigs['Vd'] + sigs['Vq']*sigs['Vq'])
    sigs['GVrms'] = 0.001 * sigs['G'] * sigs['Vrms']
    sigs['Ctl'] = impute_control_mode (sigs['t'])
    if '1ph' in filename:
      sigs['Unb'] = impute_unbalanced_mode (sigs['t'])
    else:
      sigs['Unb'] = np.zeros(len(sigs['t']))
    # create the low-pass and high-pass filtered dq0 currents
    sigs['Idlo'] = my_filter (sigs['Id'], hp=False)
    sigs['Idhi'] = my_filter (sigs['Id'], hp=True)
    sigs['Iqlo'] = my_filter (sigs['Iq'], hp=False)
    sigs['Iqhi'] = my_filter (sigs['Iq'], hp=True)
    sigs['I0lo'] = my_filter (sigs['I0'], hp=False)
    sigs['I0hi'] = my_filter (sigs['I0'], hp=True)
    for key in ['t', 'T', 'G', 'Fc', 'Md', 'Mq', 'Ctl', 'Unb', 'Vrms', 'GVrms', 'Vdc', 'Idc',
                'Idlo', 'Idhi', 'Iqlo', 'Iqhi', 'I0lo', 'I0hi']:
      grpout.create_dataset (key, data=sigs[key], compression='gzip')

print ('created {:d} training cases'.format(idx))

