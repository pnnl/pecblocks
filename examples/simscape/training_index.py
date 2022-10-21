# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: training_index.py
""" Writes a summary of the XLSX/HDF5 training case attributes.

"""

import sys
import numpy as np
import h5py

idx1 = 950
idx2 = 2450
idx2 = 1950

tags = ['G', 'T', 'Fc', 'Md', 'Mq', 'Vdc', 'Idc', 'Id', 'Iq', 'Vrms']
hdr = '     G    T    Fc    Md     Mq    Vdc    Idc    Id     Iq   Vrms'
fmt = '{:6.1f} {:4.1f} {:5.2f} {:5.3f} {:6.3f} {:6.2f} {:6.2f} {:5.2f} {:6.2f} {:6.2f}'

def summarize_group(lbl, grp):
  dlen = grp['t'].len()
  y = np.zeros(dlen)
  y1 = np.zeros(len(tags))
  y2 = np.zeros(len(tags))
  for i in range(len(tags)):
    chan = tags[i]
    grp[chan].read_direct (y)
    y1[i] = y[idx1]
    y2[i] = y[idx2]
  print (('{:4s}    '+fmt+fmt).format (str(lbl), y1[0], y1[1], y1[2], y1[3], y1[4], y1[5], y1[6], y1[7], y1[8], y1[9],
                                   y2[0], y2[1], y2[2], y2[3], y2[4], y2[5], y2[6], y2[7], y2[8], y2[9]))

filename = 'test.hdf5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

print ('Case     Values[{:d}]{:s} Values[{:d}]'.format(idx1, ''.ljust(len(hdr)-11, ' '), idx2))
print ('        {:s}{:s}'.format(hdr,hdr))
with h5py.File(filename, 'r') as f:
  ngroups = len(f.items())
  for i in range(ngroups):
    summarize_group ('{:4d}'.format(i), f[str(i)])

