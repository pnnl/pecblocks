# Copyright (C) 2018-2021 Battelle Memorial Institute
# file: HDF5TrainingPlot.py
""" Plots the ATP training simulations from HDF5 files.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py

plot_defs = [
    {'row':0, 'col':0, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
    {'row':0, 'col':2, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Md', 'title':'Md', 'ylabel':'pu'},
    {'row':0, 'col':4, 'tag':'Mq', 'title':'Mq', 'ylabel':'pu'},
    {'row':1, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Dbar', 'title':'Dbar', 'ylabel':'pu'},
    {'row':1, 'col':3, 'tag':'P', 'title':'Real Power',  'ylabel':'kW', 'scale':0.001},
    {'row':1, 'col':4, 'tag':'Q', 'title':'Reactive Power',  'ylabel':'kVAR', 'scale':0.001},
    {'row':2, 'col':0, 'tag':'Vd', 'title':'Vd',  'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Vq', 'title':'Vq',  'ylabel':'V'},
    {'row':2, 'col':2, 'tag':'Id', 'title':'Id',  'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq', 'title':'Iq',  'ylabel':'A'},
    {'row':2, 'col':4, 'tag':'I0', 'title':'I0',  'ylabel':'A'}
  ]

def start_plot(case_title):
  fig, ax = plt.subplots(3, 5, sharex = 'col', figsize=(15,6), constrained_layout=True)
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
    ax[row,col].plot (t, scale * y)

def finish_plot(ax, plot_file = None):
  for j in range(5):
    ax[2,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()

filename = 'looped.hdf5'
filename = 'new.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

ax = start_plot (filename)
with h5py.File(filename, 'r') as f:
  for grp_name, grp in f.items():
    plot_group (ax, grp)
finish_plot (ax)
