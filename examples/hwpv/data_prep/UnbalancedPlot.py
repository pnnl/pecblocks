# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: UnbalancedPlot.py
""" Plots the the merged unbalanced training set.

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

input_defs = [
    {'row':0, 'col':0, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
    {'row':0, 'col':2, 'tag':'GVrms', 'title':'GVrms',  'ylabel':'kVW/m2'},
    {'row':0, 'col':3, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
    {'row':1, 'col':0, 'tag':'Md', 'title':'Md', 'ylabel':'pu'},
    {'row':1, 'col':1, 'tag':'Mq', 'title':'Mq', 'ylabel':'pu'},
    {'row':1, 'col':2, 'tag':'Ctl', 'title':'Control Mode',  'ylabel':'[0,GFM,GFL]'},
    {'row':1, 'col':3, 'tag':'Unb', 'title':'Unbalanced Mode',  'ylabel':'[0,UNB]'}
  ]

vfilt_defs = [
  {'row':0, 'col':0, 'tag':'Vdlo', 'title':'Vd Low Pass',    'ylabel':'V'},
  {'row':0, 'col':1, 'tag':'Vqlo', 'title':'Vq Low Pass',    'ylabel':'V'},
  {'row':0, 'col':2, 'tag':'V0lo', 'title':'V0 Low Pass',    'ylabel':'V'},
#  {'row':0, 'col':3, 'tag':'Vrms', 'title':'RMS Voltage',    'ylabel':'V'},
  {'row':0, 'col':3, 'tag':'GIlo', 'title':'GIrms Low Pass', 'ylabel':'kAW/m2'},
  {'row':1, 'col':0, 'tag':'Vdhi', 'title':'Vd High Pass',   'ylabel':'V'},
  {'row':1, 'col':1, 'tag':'Vqhi', 'title':'Vq High Pass',   'ylabel':'V'},
  {'row':1, 'col':2, 'tag':'V0hi', 'title':'V0 High Pass',   'ylabel':'V'},
  {'row':1, 'col':3, 'tag':'GVlo', 'title':'GVrms Low Pass', 'ylabel':'kVW/m2'}
  ]

output_defs = [
    {'row':0, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',    'ylabel':'V'},
    {'row':0, 'col':1, 'tag':'Idlo', 'title':'Id Low Pass',  'ylabel':'A'},
    {'row':0, 'col':2, 'tag':'Iqlo', 'title':'Iq Low Pass',  'ylabel':'A'},
    {'row':0, 'col':3, 'tag':'I0lo', 'title':'I0 Low Pass',  'ylabel':'A'},
    {'row':1, 'col':0, 'tag':'Idc', 'title':'DC Current',    'ylabel':'A'},
    {'row':1, 'col':1, 'tag':'Idhi', 'title':'Id High Pass', 'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Iqhi', 'title':'Iq High Pass', 'ylabel':'A'},
    {'row':1, 'col':3, 'tag':'I0hi', 'title':'I0 High Pass', 'ylabel':'A'}
  ]

def start_plot(case_title, defs):
  fig, ax = plt.subplots(2, 4, sharex = 'col', figsize=(15,6), constrained_layout=True)
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
  for j in range(4):
    ax[1,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()
  plt.close()

filename = 'd:/data/unb3t.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
  ncases = len(f.items())
  ax = start_plot ('Inputs from {:s} ({:d} cases)'.format(filename, ncases), input_defs)
  for grp_name, grp in f.items():
    plot_group (ax, grp, input_defs)
  finish_plot (ax, 'images/UnbInputs.png')

  ax = start_plot ('Voltages from {:s} ({:d} cases)'.format(filename, ncases), vfilt_defs)
  for grp_name, grp in f.items():
    plot_group (ax, grp, vfilt_defs)
  finish_plot (ax, 'images/UnbVoltages.png')

  ax = start_plot ('Outputs from {:s} ({:d} cases)'.format(filename, ncases), output_defs)
  for grp_name, grp in f.items():
    plot_group (ax, grp, output_defs)
  finish_plot (ax, 'images/UnbOutputs.png')

