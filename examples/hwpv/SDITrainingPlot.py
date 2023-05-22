# Copyright (C) 2018-2023 Battelle Memorial Institute
# file: SDITrainingPlot.py
""" Plots the SDI training data from HDF5 files.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

plt.rcParams['savefig.directory'] = os.getcwd()

plot_defs = [
    {'row':0, 'col':0, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':1, 'tag':'Rc',   'title':'Load Resistance',   'ylabel':'Ohm'},
    {'row':0, 'col':2, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':0, 'col':3, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':0, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':1, 'col':1, 'tag':'Irms', 'title':'AC RMS Current',    'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':1, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':0, 'tag':'Vrms', 'title':'AC RMS Voltage',    'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':2, 'col':2, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':2, 'col':3, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'}
  ]

def start_plot(case_title):
  fig, ax = plt.subplots(3, 4, sharex = 'col', figsize=(15,6), constrained_layout=True)
  fig.suptitle ('Dataset: ' + case_title)
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'])
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
    if 'scale' in plot:
      scale = plot['scale']
    else:
      scale = 1.0
    if 'offset' in plot:
      offset = plot['offset']
    else:
      offset = 0.0
    ax[row,col].plot (t, scale * y + offset)

def finish_plot(ax, plot_file = None):
  for j in range(4):
    ax[2,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()

filename = 'c:/data/sdi.hdf5'
ax = start_plot (filename)
with h5py.File(filename, 'r') as f:
  for grp_name, grp in f.items():
    plot_group (ax, grp)
finish_plot (ax)
