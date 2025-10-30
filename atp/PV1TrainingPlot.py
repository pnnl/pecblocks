# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: PV1TrainingPlot.py
""" Plots the ATP PV1 training simulations from HDF5 files.

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
 
nrows = 3
ncols = 5
#tticks = [0.0,0.4,0.8,1.2,1.6,2.0]
#tticks = [0.0,0.5,1.0,1.5,2.0,2.5,3.5]
#tticks = [0.0,2.0,4.0,6.0,8.0]
tticks = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]
tmin = tticks[0]
tmax = tticks[-1]
 
plot_defs = [
    {'row':0, 'col':0, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
    {'row':0, 'col':1, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
    {'row':0, 'col':2, 'tag':'Ud', 'title':'Control Ud', 'ylabel':'pu'},
    {'row':0, 'col':3, 'tag':'Uq', 'title':'Control Uq', 'ylabel':'pu'},
    {'row':0, 'col':4, 'tag':'Fc', 'title':'Control Frequency', 'ylabel':'Hz'},
    {'row':1, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V'},
    {'row':1, 'col':1, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A'},
    {'row':1, 'col':2, 'tag':'Vrms', 'title':'AC Voltage', 'ylabel':'V rms'},
    {'row':1, 'col':3, 'tag':'Irms', 'title':'AC Current', 'ylabel':'A rms'},
    {'row':1, 'col':4, 'tag':'Ppvpu', 'title':'Panel Power', 'ylabel':'pu'},
    {'row':2, 'col':0, 'tag':'Vd', 'title':'Vd Output',  'ylabel':'V'},
    {'row':2, 'col':1, 'tag':'Vq', 'title':'Vq Output', 'ylabel':'A'},
    {'row':2, 'col':2, 'tag':'Id', 'title':'Id Output', 'ylabel':'A'},
    {'row':2, 'col':3, 'tag':'Iq', 'title':'Iq Output', 'ylabel':'A'},
    {'row':2, 'col':4, 'tag':'D', 'title':'MPPT D Ratio',  'ylabel':'pu'},
  ]

def start_plot(case_title):
  fig, ax = plt.subplots(nrows, ncols, sharex = 'col', figsize=(15,8), constrained_layout=True)
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
    ax[row,col].plot (t, y)

def finish_plot(ax, plot_file = None):
  for j in range(ncols):
    for i in range(nrows):
      ax[i,j].grid ()
      ax[i,j].set_xlim (tmin, tmax)
      ax[i,j].set_xticks (tticks)
    ax[nrows-1,j].set_xlabel ('Seconds')
  if plot_file:
    plt.savefig(plot_file)
  plt.show()
  plt.close()

filename = 'osg.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

ax = start_plot (filename)
with h5py.File(filename, 'r') as f:
#  plot_group (ax, f['group0']) # 32
#  finish_plot (ax)
#  quit()

  for grp_name, grp in f.items():
    plot_group (ax, grp)

finish_plot (ax)
