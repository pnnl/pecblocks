# Copyright (C) 2018-2025 Battelle Memorial Institute
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

lsize = 14
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=lsize)
plt.rc('ytick', labelsize=lsize)
plt.rc('axes', labelsize=lsize)
plt.rc('legend', fontsize=lsize)
pWidth = 6.0
pHeight = pWidth / 1.618

plot_defs = [ # ucf2t
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',    'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial GVrms',  'ylabel':''},
    {'row':1, 'col':0, 'tag':'Md1',  'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Mq1',  'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vod',  'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Voq',  'title':'Vq',                'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'GIrms','title':'Polynomial GIrms',  'ylabel':''},
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

# this is for the Feb 1 and March 11, 2024 data sets, without Fc, T, and Rload
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

# this is for the May 8, 2024 data sets, augmented with GIrms, Vd and Vq outputs: ucf9, ucf9c
plot_defs = [
    {'row':0, 'col':0, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':2, 'tag':'GVrms','title':'Polynomial GVrms',  'ylabel':''},
    {'row':0, 'col':3, 'tag':'GIrms','title':'Polynomial GIrms',  'ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':1, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':2, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'}
  ]

# this is for the May 23, 2024 data sets, augmented with GIrms and GVrms, Step, Ramp: ucf4
plot_defs = [
    {'row':0, 'col':0, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':2, 'tag':'Step', 'title':'Control Step',      'ylabel':''},
    {'row':0, 'col':3, 'tag':'Ramp', 'title':'Control Ramp',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'Rload', 'title':'$R_{load}$',            'ylabel':'$\Omega$'},

    {'row':1, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Vd',   'title':'$V_d$',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vq',   'title':'$V_q$',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Ud',   'title':'$U_d$',                'ylabel':''},
    {'row':1, 'col':4, 'tag':'GVrms','title':'Polynomial $G_{Vrms}$',  'ylabel':''},

    {'row':2, 'col':0, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':1, 'tag':'Id',   'title':'$I_d$',                'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Iq',   'title':'$I_q$',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Uq',   'title':'$U_q$',                'ylabel':''},
    {'row':2, 'col':4, 'tag':'GIrms','title':'Polynomial $G_{Irms}$',  'ylabel':''}
  ]

# this is for the June 3, 2025 data set for sigma optimization
plot_defs = [
    {'row':0, 'col':0, 'tag':'Step', 'title':'Control Step', 'ylabel':''},
    {'row':0, 'col':1, 'tag':'Ramp', 'title':'Control Ramp', 'ylabel':''},
    {'row':0, 'col':3, 'tag':'Ra',   'title':'$R_a$',        'ylabel':'$\Omega$'},

    {'row':1, 'col':0, 'tag':'Rb',   'title':'$R_b$', 'ylabel':'$\Omega$'},
    {'row':1, 'col':1, 'tag':'Rc',   'title':'$R_c$', 'ylabel':'$\Omega$'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'$V_d$', 'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'$V_q$', 'ylabel':'V'},

    {'row':2, 'col':0, 'tag':'Ud',   'title':'$U_d$', 'ylabel':''},
    {'row':2, 'col':1, 'tag':'Uq',   'title':'$U_q$', 'ylabel':''},
    {'row':2, 'col':2, 'tag':'Id',   'title':'$I_d$', 'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'$I_q$', 'ylabel':'A'}
  ]

def start_plot(case_title, idx):
  last_col = 0
  last_row = 0
  for plot in plot_defs:
    if plot['row'] > last_row:
      last_row = plot['row']
    if plot['col'] > last_col:
      last_col = plot['col']
  fig, ax = plt.subplots(last_row + 1, last_col + 1, sharex = 'col', figsize=(3*(last_col+1),2.5*(last_row+1)), constrained_layout=True)
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
  last_row = len(ax) - 1
  n_col = len(ax[0])
  for j in range(n_col):
    ax[last_row,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()


#pathname = 'd:/data/'
#pathname = 'd:/data/ucf3/'
pathname = 'd:/data/ucf_sigma/'

if __name__ == '__main__':
  idx = -1
  if len(sys.argv) > 1:
    idx = int(sys.argv[1])
  for root in ['ucf_s1']: # ucf4, ucf3z. ucf2t, ucf3, ucf7, ucf9, ucf9c
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
