# Copyright (C) 2021 Battelle Memorial Institute
# file: h5view.py
""" Show structure of a HDF5 file.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import h5py
import numpy as np

#fname = 'gfm.hdf5'
fname = 'looped.hdf5'
if len(sys.argv) > 1:
  fname = sys.argv[1]

with h5py.File(fname, 'r') as f:
  print ('there are', len(f.items()), 'groups and the first one is:')
  for grp_name, grp in f.items():
    print (' ', grp_name)
    for key in grp:
      print ('   ', key)

    dlen = grp['t'].len()
    t = np.zeros(dlen)
    grp['t'].read_direct (t)
    dt = t[1] - t[0]
    print (' {:d} time points from {:.7f} to {:.7f} at step={:.7f}'.format(dlen, t[0], t[-1], dt))
    quit()

