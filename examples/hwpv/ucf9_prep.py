# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: ucf9_prep.py
""" Reduces unused inputs from the UCF3x Simscape 1-ms training data.
"""

import numpy as np
import h5py

PREFIX = 'ucf'
input_path = 'd:/data/ucf3/ucf3x.hdf5'
output_path = 'd:/data/ucf3/ucf9.hdf5'

if __name__ == '__main__':
  print ('Copying UCF3 training records from {:s} to {:s}'.format (input_path, output_path))
  f_out = h5py.File (output_path, 'w')

  idx = 0
  n = -1

  print ('Copying simulation records')
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

# print ('Augmentation starts at {:s}{:d}, n={:d}, tmax={:.5f}'.format (PREFIX, idx, n, tmax))
# zeros = np.zeros(n)
# ones = np.ones(n)
# tbase = np.linspace (0.0, tmax, n)
# for ctrl in [0.0, 1.0]:
#   for Ud in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
#     for Uq in [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
#       new_name = '{:s}{:d}'.format (PREFIX, idx)
#       idx += 1
#       grp = f_out.create_group (new_name)
#       grp.create_dataset ('t', data=tbase, compression='gzip')
#       for tag in ['G', 'Vd', 'Vq', 'GVrms', 'Idc', 'Vdc', 'Id', 'Iq']:
#         grp.create_dataset (tag, data=zeros, compression='gzip')
#       grp.create_dataset ('Ctrl', data=ctrl*ones, compression='gzip')
#       grp.create_dataset ('Ud', data=Ud*ones, compression='gzip')
#       grp.create_dataset ('Uq', data=Uq*ones, compression='gzip')

  f_out.close()
  print ('Augmentation ends at {:s}'.format(new_name))

