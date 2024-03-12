# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: ucf3_prep.py
""" Augments the UCF3 Simscape training data with sone zero-output records.
"""

import numpy as np
import h5py
import os

PREFIX = 'ucf'
path_name = 'd:/data/ucf3/'
inputs = ['ucf3.hdf5', 'ucf3i.hdf5']
output = 'ucf7.hdf5'

if __name__ == '__main__':
  print ('Augmenting UCF3 training records in {:s} from {:s} to {:s}'.format (path_name, str(inputs), output))
  f_out = h5py.File (os.path.join(path_name, output), 'w')

  idx = 0
  n = -1

  print ('Copying simulation records')
  for fname in inputs:
    input_path = os.path.join (path_name, fname)
    with h5py.File(input_path, 'r') as f_in:
      for grp_name, grp_in in f_in.items():
        if n < 0:
          n = grp_in['t'].len()
          tbase = np.zeros(n)
          grp_in['t'].read_direct (tbase)
          tmax = tbase[-1]
        new_name = '{:s}{:d}'.format (PREFIX, idx)
        idx += 1
        grp_out = f_out.create_group (new_name)
        for tag in ['t', 'G', 'Ctrl', 'Ud', 'Uq', 'Vdc', 'Idc', 'Vd', 'Vq', 'GVrms', 'Id', 'Iq']:
          grp_out.create_dataset (tag, data=grp_in[tag], compression='gzip')

  f_out.close()
  print ('Augmentation ends at {:s}'.format(new_name))

