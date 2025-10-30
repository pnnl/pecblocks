# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: Augment1ph.py
""" Processes time-varying dq0 signals into steady components for training.

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

plot_defs = [
    {'row':0, 'col':0, 'tag':'Vd', 'title':'Vd',  'ylabel':'V'},
    {'row':0, 'col':1, 'tag':'Vq', 'title':'Vq',  'ylabel':'V'},
    {'row':0, 'col':2, 'tag':'V0', 'title':'V0',  'ylabel':'V'},
    {'row':0, 'col':3, 'tag':'Id', 'title':'Id',  'ylabel':'A'},
    {'row':0, 'col':4, 'tag':'Iq', 'title':'Iq',  'ylabel':'A'},
    {'row':0, 'col':5, 'tag':'I0', 'title':'I0',  'ylabel':'A'}
  ]

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

def start_plot(case_title, ncases):
  fig, ax = plt.subplots(3, 6, sharex = 'col', figsize=(18,9), constrained_layout=True)
  fig.suptitle ('Dataset: {:s} ({:d} cases)'.format(case_title, ncases))
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])

    plt_ax = ax[plot['row']+1, plot['col']]
    plt_ax.set_title ('Low Pass')
    plt_ax.set_ylabel (plot['ylabel'])

    plt_ax = ax[plot['row']+2, plot['col']]
    plt_ax.set_title ('High Pass Average')
    plt_ax.set_ylabel (plot['ylabel'])
  return ax

def plot_group(ax, grp):
  dlen = grp['t'].len()
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  for plot in plot_defs:
    row = plot['row']
    col = plot['col']
    grp[plot['tag']].read_direct (y)
#    avg = moving_average (y)
    lp = my_filter (y)
    hp = my_filter (y, hp=True)
    ax[row,col].plot (t, y)
    ax[row+1,col].plot (t, lp)
    ax[row+2,col].plot (t, hp)

def finish_plot(ax, plot_file = None):
  for j in range(6):
    ax[2,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()

filename = 'looped_hw8_1ph.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
  ax = start_plot (filename, len(f.items()))
  for grp_name, grp in f.items():
    plot_group (ax, grp)
    # break
  finish_plot (ax)
