# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: pv3_augment.py
""" Plots the ATP training simulations from HDF5 files.

Augments the channels with Ctl, Vrms, GVrms.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math

ctl_t = [0.00, 1.00, 1.01, 10.00]
ctl_y = [0.00, 0.00, 1.00, 1.00]

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
    {'row':3, 'col':0, 'tag':'Vrms', 'title':'Vrms',  'ylabel':'V'},
    {'row':3, 'col':1, 'tag':'GVrms', 'title':'GVrms', 'ylabel':''},
    {'row':3, 'col':2, 'tag':'Ctl', 'title':'Ctl Mode',  'ylabel':''},
    {'row':3, 'col':3, 'tag':'Pdc', 'title':'DC Power',  'ylabel':'kW', 'scale':0.001}
  ]

def start_plot(case_title):
  fig, ax = plt.subplots(4, 5, sharex = 'col', figsize=(15,9), constrained_layout=True)
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

filename = 'data/gfm8.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

ax = start_plot (filename)
with h5py.File(filename, 'a') as f:
  for grp_name, grp in f.items():
    dlen = grp['t'].len()
    if 'Vrms' not in grp:
      print ('creating Vrms for', grp_name)
      vd = np.zeros(dlen)
      vq = np.zeros(dlen)
      grp['Vd'].read_direct(vd)
      grp['Vq'].read_direct(vq)
      vrms = math.sqrt(1.5) * np.sqrt(vd*vd + vq*vq)
      grp.create_dataset ('Vrms', data=vrms, compression='gzip')
    if 'GVrms' not in grp:
      print ('creating GVrms for', grp_name)
      Vrms = np.zeros(dlen)
      G = np.zeros(dlen)
      grp['Vrms'].read_direct(Vrms)
      grp['G'].read_direct(G)
      gvrms = 0.001 * G * Vrms
      grp.create_dataset ('GVrms', data=gvrms, compression='gzip')
    if 'Pdc' not in grp:
      print ('creating Pdc for', grp_name)
      Vdc = np.zeros(dlen)
      Idc = np.zeros(dlen)
      grp['Vdc'].read_direct(Vdc)
      grp['Idc'].read_direct(Idc)
      pdc = Vdc * Idc
      grp.create_dataset ('Pdc', data=pdc, compression='gzip')
    if 'Ctl' not in grp:
      print ('creating Ctl for', grp_name)
      t = np.zeros(dlen)
      grp['t'].read_direct(t)
      ctl = np.zeros(dlen)
      for i in range(dlen):
        ctl[i] = np.interp (t[i], ctl_t, ctl_y)
      grp.create_dataset ('Ctl', data=ctl, compression='gzip')
    plot_group (ax, grp)
finish_plot (ax)
