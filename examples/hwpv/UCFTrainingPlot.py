# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: UCFTrainingPlot.py
""" Plots the UCF Simscape training data from HDF5 files.

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
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',    'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Md1',  'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Mq1',  'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vod',  'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Voq',  'title':'Vq',                'ylabel':'V'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'}
  ]

# this is for the February 1, 2024 data set
plot_defs = [
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'Rload','title':'Rload (Unused)',    'ylabel':'Ohm'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'}
  ]

# this is for the February 1, 2024 data set, without Fc, T, and Rload
plot_defs = [
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':3, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'}
  ]

def start_plot(case_title, idx):
  fig, ax = plt.subplots(3, 5, sharex = 'col', figsize=(15,6), constrained_layout=True)
  if idx < 0:
    fig.suptitle ('Dataset: ' + case_title)
  else:
    fig.suptitle ('Dataset: ' + case_title + ' Group: ' + str(idx))
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])
  return ax

def plot_group(ax, grp):
  dlen = grp['t'].len()
#  print (dlen, 'points')
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


pathname = 'd:/data/'
pathname = 'd:/data/ucf3/'

if __name__ == '__main__':
  idx = -1
  if len(sys.argv) > 1:
    idx = int(sys.argv[1])
  for root in ['ucf3z']: # ucf2, ucf3
    filename = '{:s}{:s}.hdf5'.format (pathname, root)
    pngname = '{:s}_Training_Set.png'.format (root)
    ax = start_plot (filename, idx)
    with h5py.File(filename, 'r') as f:
      if idx >= 0:
        pngname = '{:s}_Group_{:d}.png'.format (root, idx)
        plot_group (ax, f['ucf{:d}'.format (idx)]) # f[str(idx)])
      else:
        pngname = '{:s}_Training_Set.png'.format (root)
        for grp_name, grp in f.items():
          plot_group (ax, grp)
  finish_plot (ax, pngname)
