# Copyright (C) 2021-22 Battelle Memorial Institute
# file: h5view.py
# shows structure of an HDF5 file
#  arg1: input file name, default test.hdf5
#  example: python h5view.py new.hdf5
#
# Assumes the HDF5 stores COMTRADE analog channels, which
# are compatible with HWPV training scripts.

import sys
import h5py
import numpy as np

fname = 'test.hdf5'
if len(sys.argv) > 1:
  fname = sys.argv[1]

with h5py.File(fname, 'r') as f:
  for grp_name, grp in f.items():
    print ('there are', len(f.items()), 'groups and the first one has {:d} items:'.format (len(grp)))
    print (' ', grp_name)
    for key in grp:
      print ('   ', key)
    break

  dlen = grp['t'].len()
  t = np.zeros(dlen)
  grp['t'].read_direct (t)
  dt = t[1] - t[0]
  print (' {:d} time points from {:.6f} to {:.6f} at step={:.6f}'.format(dlen, t[0], t[-1], dt))

