# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: TrainingDataPlot.py
""" Plots the HWPV training data from HDF5 files.
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

plot_defs = {
  "big3":[
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Md',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Mq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'}
  ],
  "lab2":[
    {'row':0, 'col':0, 'tag':'Rc',   'title':'Load Resistance',   'ylabel':'Ohm'},
    {'row':0, 'col':1, 'tag':'Vc',   'title':'Control Voltage',   'ylabel':'V'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':1, 'col':0, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':1, 'col':1, 'tag':'Irms', 'title':'AC RMS Current',    'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':1, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':0, 'tag':'Vrms', 'title':'AC RMS Voltage',    'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':2, 'col':2, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'}
  ],
  "osg4":[
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':1, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':2, 'col':0, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':1, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':2, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'}
  ],
  "sdi5":[
    {'row':0, 'col':0, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':1, 'tag':'Rc',   'title':'Load Resistance',   'ylabel':'Ohm'},
    {'row':0, 'col':2, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':0, 'col':3, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Vrms', 'title':'AC RMS Voltage',    'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Irms', 'title':'AC RMS Current',    'ylabel':'A'},
    {'row':2, 'col':0, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':1, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':2, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'}
  ],
  "ucf2":[
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Md1',  'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Mq1',  'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vod',  'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Voq',  'title':'Vq',                'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'Isd',  'title':'Bridge Id',         'ylabel':'A'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':4, 'tag':'Isq',  'title':'Bridge Iq',         'ylabel':'A'}
  ],
  "ucf3":[
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'GVrms','title':'Polynomial Feature','ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'Rload','title':'Load Resistance',   'ylabel':'Ohm'},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
  ],
  "ucf4":[
    {'row':0, 'col':0, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'Rload','title':'Load Resistance',   'ylabel':'Ohm'},
    {'row':0, 'col':2, 'tag':'GVrms','title':'Polynomial GVrms',  'ylabel':''},
    {'row':0, 'col':3, 'tag':'GIrms','title':'Polynomial GIrms',  'ylabel':''},
    {'row':0, 'col':4, 'tag':'Ctrl', 'title':'Control Mode',      'ylabel':''},
    {'row':1, 'col':0, 'tag':'Ud',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Uq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':1, 'col':4, 'tag':'Ramp', 'title':'Control Ramp',      'ylabel':''},
    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':2, 'col':4, 'tag':'Step', 'title':'Control Step',      'ylabel':''}
  ],
  "unb3":[
    {'row':0, 'col':0, 'tag':'T',    'title':'Temperature',       'ylabel':'C'},
    {'row':0, 'col':1, 'tag':'G',    'title':'Irradiance',        'ylabel':'W/m2'},
    {'row':0, 'col':2, 'tag':'Fc',   'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':0, 'col':3, 'tag':'Ctl',  'title':'Control Mode',      'ylabel':''},
    {'row':0, 'col':4, 'tag':'Unb',  'title':'Unbalance Mode',    'ylabel':''},

    {'row':1, 'col':0, 'tag':'Md',   'title':'Ud',                'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Mq',   'title':'Uq',                'ylabel':'V'},
    {'row':1, 'col':2, 'tag':'Vrms', 'title':'AC RMS Voltage',    'ylabel':'V'},
    {'row':1, 'col':3, 'tag':'GVrms','title':'Polynomial GVrms',  'ylabel':''},
    {'row':1, 'col':4, 'tag':'GVlo', 'title':'GVrms Low',         'ylabel':''},

    {'row':2, 'col':0, 'tag':'Vdc',  'title':'DC Voltage',        'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Idc',  'title':'DC Current',        'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'I0',   'title':'I0',                'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'I0lo', 'title':'I0 Low',            'ylabel':'A'},
    {'row':2, 'col':4, 'tag':'I0hi', 'title':'I0 High',           'ylabel':'A'},

    {'row':3, 'col':0, 'tag':'Vd',   'title':'Vd',                'ylabel':'V'},
    {'row':4, 'col':0, 'tag':'Vdlo', 'title':'Vd Low',            'ylabel':'V'},
    {'row':5, 'col':0, 'tag':'Vdhi', 'title':'Vd High',           'ylabel':'V'},

    {'row':3, 'col':1, 'tag':'Vq',   'title':'Vq',                'ylabel':'V'},
    {'row':4, 'col':1, 'tag':'Vqlo', 'title':'Vq Low',            'ylabel':'V'},
    {'row':5, 'col':1, 'tag':'Vqhi', 'title':'Vq High',           'ylabel':'V'},

    {'row':3, 'col':2, 'tag':'V0',   'title':'V0',                'ylabel':'V'},
    {'row':4, 'col':2, 'tag':'V0lo', 'title':'V0 Low',            'ylabel':'V'},
    {'row':5, 'col':2, 'tag':'V0hi', 'title':'V0 High',           'ylabel':'V'},

    {'row':3, 'col':3, 'tag':'Id',   'title':'Id',                'ylabel':'A'},
    {'row':4, 'col':3, 'tag':'Idlo', 'title':'Id Low',            'ylabel':'A'},
    {'row':5, 'col':3, 'tag':'Idhi', 'title':'Id High',           'ylabel':'A'},

    {'row':3, 'col':4, 'tag':'Iq',   'title':'Iq',                'ylabel':'A'},
    {'row':4, 'col':4, 'tag':'Iqlo', 'title':'Iq Low',            'ylabel':'A'},
    {'row':5, 'col':4, 'tag':'Iqhi', 'title':'Iq High',           'ylabel':'A'}
  ]
}

def start_plot (plt_def, case_title, idx):
  nrows = 0 # find the highest row and column index used
  ncols = 0
  for plot in plt_def:
    if plot['row'] > nrows:
      nrows = plot['row']
    if plot['col'] > ncols:
      ncols = plot['col']
  nrows += 1
  ncols += 1
  fig, ax = plt.subplots(nrows, ncols, sharex = 'col', figsize=(15,2*nrows), constrained_layout=True)
  if case_title is not None:
    if idx < 0:
      fig.suptitle ('Dataset: ' + case_title)
    else:
      fig.suptitle ('Dataset: ' + case_title + ' Group: ' + str(idx))
  for plot in plt_def:
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'], fontsize=lsize)
    plt_ax.set_ylabel (plot['ylabel'])
  return ax

def plot_group (plt_def, ax, grp, linewidth=1.0):
  dlen = grp['t'].len()
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  for plot in plt_def:
    if 'tag' in plot:
      grp[plot['tag']].read_direct (y)
      row = plot['row']
      col = plot['col']
      ax[row,col].plot (t, y, linewidth=linewidth)

def finish_plot (plt_def, ax, plot_file = None, add_constants = False):
  if add_constants:
    ax[0, 0].plot ([0.0, 4.0], [60.0, 60.0], label='Fc [Hz]')
    ax[0, 0].plot ([0.0, 4.0], [35.0, 35.0], label='T [C]')
    ax[0, 0].legend ()
  for j in range(4):
    ax[2,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()
  plt.close()

pathname = './'

if __name__ == '__main__':
  root = 'ucf2'
  if len(sys.argv) > 1:
    root = sys.argv[1]
  idx = -1

  filename = '{:s}{:s}.hdf5'.format (pathname, root)
  pngname = '{:s}_Training_Set.png'.format (root)
  plt_def = plot_defs[root]
  ax = start_plot (plt_def, case_title=None, idx=idx)
  with h5py.File(filename, 'r') as f:
    if idx >= 0:
      pngname = '{:s}_Group_{:d}.png'.format (root, idx)
      plot_group (plt_def, ax, f['ucf{:d}'.format (idx)]) # f[str(idx)])
    else:
      pngname = '{:s}_Training_Set.png'.format (root)
      for grp_name, grp in f.items():
        plot_group (plt_def, ax, grp, linewidth=0.5)
  finish_plot (plt_def, ax, plot_file = pngname, add_constants=False) # add_constants=True for SysDO
