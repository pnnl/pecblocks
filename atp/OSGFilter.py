# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: OSGFilter.py
""" Plots the the merged unbalanced training set.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal
import os

plt.rcParams['savefig.directory'] = os.getcwd()

input_defs = [
    {'row':0, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V'},
    {'row':0, 'col':1, 'tag':'Vq', 'title':'Vq',  'ylabel':'V'},
  ]

def start_plot(case_title, defs):
  maxrow = 0
  maxcol = 0
  for plot in defs:
    if plot['row'] > maxrow:
      maxrow = plot['row']
    if plot['col'] > maxcol:
      maxcol = plot['col']
  fig, ax = plt.subplots(maxrow+1, maxcol+1, sharex = 'col', figsize=(15,6), constrained_layout=True)
  fig.suptitle ('Dataset: ' + case_title)
  for plot in defs:
    plt_ax = ax[plot['col']]
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])
  return ax

ndec=400
b, a = signal.butter (2, 1.0 / 4096.0, btype='lowpass', analog=False)

def plot_group(ax, grp, defs):
  dlen = grp['t'].len()
  print (dlen, 'raw samples')
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  for plot in defs:
    row = plot['row']
    col = plot['col']
    grp[plot['tag']].read_direct (y)
    ax[col].plot (t, y, color='blue', label='raw')

    ydec = signal.lfilter (b, a, y)[::ndec]
    tdec = t[::ndec]

    ax[col].plot (tdec, ydec, color='red', label='dec')
  print (len(tdec), 'dec samples at dt={:.7f}'.format(tdec[1]-tdec[0]))

def finish_plot(ax, plot_file = None):
  for j in range(ax.shape[0]):
    ax[j].set_xlabel ('Seconds')
    ax[j].legend()
    ax[j].grid()
  if plot_file:
    plt.savefig(plot_file)
  plt.show()

filename = 'osg.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
  ncases = len(f.items())
  grp_name = 'group32'
  ax = start_plot ('Inputs from {:s} group {:s} ({:d} cases)'.format(filename, grp_name, ncases), input_defs)
#  for grp_name, grp in f.items():
#    print (grp_name)
  plot_group (ax, f[grp_name], input_defs)
  finish_plot (ax)

