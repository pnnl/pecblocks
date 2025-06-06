# Copyright (C) 2018-2025 Battelle Memorial Institute
# file: ucf_sigma_prep.py
""" Augment signals from the UCF 6/3/25 Simscape 10-ms training data. 
    Creates Step and Ramp versions of Ctrl signal, for Norton sigma-optimized model.
    Rename Isd and Isq to Id and Iq.
"""

import numpy as np
import h5py

PREFIX = 'ucf'
input_path = 'd:/data/ucf_sigma/new_data_model_ucf.hdf5'
output_path = 'd:/data/ucf_sigma/ucf_s1.hdf5'
GFM_TIME = 1.0
DT = 0.01

if __name__ == '__main__':
  print ('Augmenting UCF4 training records from {:s} to {:s}'.format (input_path, output_path))
  f_out = h5py.File (output_path, 'w')

  idx = 0
  n = -1

  print ('Copying and augmenting simulation records')
  with h5py.File(input_path, 'r') as f_in:
    ncases = len(f_in.items())
    for grp_name, grp_in in f_in.items():
      if idx % 20 == 0:
        print ('building {:d} of {:d} records'.format (idx, ncases))
      if n < 0:
        n = grp_in['t'].len()
        tbase = np.zeros(n)
        grp_in['t'].read_direct (tbase)
        tmax = tbase[-1]
        step = np.interp(tbase, [0.0, GFM_TIME - DT, GFM_TIME, 100.0], [0.0, 0.0, 1.0, 1.0])
        ramp = np.interp(tbase, [0.0, GFM_TIME, 100.0], [0.0, 1.0, 1.0])
      new_name = '{:s}{:d}'.format (PREFIX, idx)
      idx += 1

      grp_out = f_out.create_group (new_name)
      grp_out.create_dataset ('Step', data=step, compression='gzip')
      grp_out.create_dataset ('Ramp', data=ramp, compression='gzip')
      grp_out.create_dataset ('Id', data=grp_in['Isd'], compression='gzip')
      grp_out.create_dataset ('Iq', data=grp_in['Isq'], compression='gzip')
      for tag in ['t', 'Ud', 'Uq', 'Ra', 'Rb', 'Rc', 'Vd', 'Vq']:
        grp_out.create_dataset (tag, data=grp_in[tag], compression='gzip')

  f_out.close()
  print ('Augmentation ends at {:s}'.format(new_name))

