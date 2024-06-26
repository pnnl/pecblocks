# Copyright (C) 2022-24 Battelle Memorial Institute

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

vrms_defs = [
  {'row':0, 'col':0, 'tag':'T', 'title':'Temperature', 'ylabel':'C'},
  {'row':0, 'col':1, 'tag':'G', 'title':'Irradiance',  'ylabel':'W/m2'},
  {'row':0, 'col':2, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
  {'row':0, 'col':3, 'tag':'Ud', 'title':'Ud', 'ylabel':'pu'},
  {'row':1, 'col':0, 'tag':'Uq', 'title':'Uq', 'ylabel':'pu'},
  {'row':1, 'col':1, 'tag':'Vrms', 'title':'Vrms',  'ylabel':'V'},
  {'row':1, 'col':2, 'tag':'GVrms', 'title':'GVrms',  'ylabel':'kVW/m2'},
  {'row':1, 'col':3, 'tag':'Ctl', 'title':'Control Mode',  'ylabel':'[0,GFM,GFL]'},
  {'row':2, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V', 'color':'red'},
  {'row':2, 'col':1, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A', 'color':'red'},
  {'row':2, 'col':2, 'tag':'Id',  'title':'Id',          'ylabel':'A', 'color':'red'},
  {'row':2, 'col':3, 'tag':'Iq',  'title':'Iq',          'ylabel':'A', 'color':'red'}
]

vdq_defs = [
  {'row':0, 'col':0, 'tag':'T',  'title':'Temperature', 'ylabel':'C'},
  {'row':0, 'col':1, 'tag':'G',  'title':'Irradiance',  'ylabel':'W/m2'},
  {'row':0, 'col':2, 'tag':'Fc', 'title':'Control Frequency',  'ylabel':'Hz'},
  {'row':0, 'col':3, 'tag':'Ctl', 'title':'Control Mode',  'ylabel':'[0,GFM,GFL]'},
  {'row':1, 'col':0, 'tag':'Ud', 'title':'Control Voltage', 'ylabel':'pu'},
  {'row':1, 'col':0, 'tag':'Uq', 'title':'Control Voltage', 'ylabel':'pu', 'color':'green','legend':True},
  {'row':1, 'col':1, 'tag':'GVrms', 'title':'GVrms',  'ylabel':'kVW/m2'},
  {'row':1, 'col':2, 'tag':'Vd',  'title':'Vd', 'ylabel':'V'},
  {'row':1, 'col':3, 'tag':'Vq',  'title':'Vq',  'ylabel':'V'},
  {'row':2, 'col':0, 'tag':'Vdc', 'title':'DC Voltage',  'ylabel':'V', 'color':'red'},
  {'row':2, 'col':1, 'tag':'Idc', 'title':'DC Current',  'ylabel':'A', 'color':'red'},
  {'row':2, 'col':2, 'tag':'Id',  'title':'Id',          'ylabel':'A', 'color':'red'},
  {'row':2, 'col':3, 'tag':'Iq',  'title':'Iq',          'ylabel':'A', 'color':'red'}
]

def plot_group(ax, grp, defs):
  dlen = grp['t'].len()
  t = np.zeros(dlen)
  y = np.zeros(dlen)
  grp['t'].read_direct (t)
  for plot in defs:
    row = plot['row']
    col = plot['col']
    color = 'blue'
    if 'color' in plot:
      color = plot['color']
    grp[plot['tag']].read_direct (y)
    if 'scale' in plot:
      scale = plot['scale']
    else:
      scale = 1.0
    ax[row,col].plot (t, scale * y, color=color, label=plot['tag'])
    ax[row,col].set_title (plot['title'])
    ax[row,col].set_ylabel (plot['ylabel'])
    if 'legend' in plot:
      ax[row,col].legend()

if __name__ == '__main__':
  plt.rcParams['savefig.directory'] = os.getcwd()
  fname = 'hwpv_pi.hdf5'

  with h5py.File(fname, 'r') as f:
    for grp_name, grp in f.items():
      if 'Vrms' in grp:
        plot_defs = vrms_defs
      else:
        plot_defs = vdq_defs
      fig, ax = plt.subplots(3, 4, sharex = 'col', figsize=(15,6), constrained_layout=True)
      fig.suptitle ('Dataset: {:s}, Group: {:s}'.format (fname, grp_name))
      plot_group (ax, grp, plot_defs)

  for j in range(4):
    ax[2,j].set_xlabel ('Seconds')
  plt.show()

