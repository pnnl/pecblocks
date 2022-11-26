# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: training_plot.py
""" Plots the the unmerged balanced training set.

This script verifies that test.hdf5, as created by
xlsx_to_hdf5.py, is in the format expected by the
HWPV traning scripts. If test.hdf5 contains multiple
cases, all will be superimposed on the same graphs.

"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py

input_defs = [
    {'row':0, 'col':0, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
    {'row':0, 'col':2, 'tag':'Vrms', 'title':'Vrms',  'ylabel':'V'},
    {'row':0, 'col':3, 'tag':'GVrms', 'title':'GVrms',  'ylabel':'kVW/m2'},
    {'row':0, 'col':4, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
    {'row':1, 'col':0, 'tag':'Md1', 'title':'Md', 'ylabel':'pu'},
    {'row':1, 'col':1, 'tag':'Mq1', 'title':'Mq', 'ylabel':'pu'},
    {'row':1, 'col':2, 'tag':'Vod', 'title':'Vd', 'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Voq', 'title':'Vq', 'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'Ctl', 'title':'Control Mode',  'ylabel':'[0,GFM,GFL]'},
  ]

output_defs = [
    {'row':0, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V'},
    {'row':0, 'col':1, 'tag':'Id',  'title':'Id',          'ylabel':'A'},
    {'row':1, 'col':0, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A'},
    {'row':1, 'col':1, 'tag':'Iq',  'title':'Iq',          'ylabel':'A'},
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
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])
  return ax

def plot_group(ax, grp, defs):
  dlen = grp['t'].len()
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  for plot in defs:
    row = plot['row']
    col = plot['col']
    grp[plot['tag']].read_direct (y)
    if 'scale' in plot:
      scale = plot['scale']
    else:
      scale = 1.0
    ax[row,col].plot (t, scale * y)

def finish_plot(ax, plot_file = None):
  for j in range(ax.shape[1]):
    ax[1,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()

filename = 'test.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
  ax = start_plot ('Inputs from {:s}'.format(filename), input_defs)
  for grp_name, grp in f.items():
    plot_group (ax, grp, input_defs)
  finish_plot (ax, 'Ucf2Inputs.png')

  ax = start_plot ('Outputs from {:s}'.format(filename), output_defs)
  for grp_name, grp in f.items():
    plot_group (ax, grp, output_defs)
  finish_plot (ax, 'Ucf2Outputs.png')

