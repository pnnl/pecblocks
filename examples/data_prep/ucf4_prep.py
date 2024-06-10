# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: ucf4_prep.py
""" Augment signals from the UCF 5/23/24 Simscape 10-ms training data. 
    Creates Step and Ramp versions of Ctrl signal, creates GIrms and GVrms.
"""

import numpy as np
import h5py
import math

PREFIX = 'ucf'
input_path = 'd:/data/ucf3/ucf4raw.hdf5'
output_path = 'd:/data/ucf3/ucf4.hdf5'
KRMS = math.sqrt(1.5)
GFM_TIME = 1.9
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
      G = np.zeros(n)
      Id = np.zeros(n)
      Iq = np.zeros(n)
      Vd = np.zeros(n)
      Vq = np.zeros(n)
      grp_in['G'].read_direct (G)
      grp_in['Id'].read_direct (Id)
      grp_in['Iq'].read_direct (Iq)
      grp_in['Vd'].read_direct (Vd)
      grp_in['Vq'].read_direct (Vq)
      GIrms = G * KRMS * np.sqrt (Id*Id + Iq*Iq)
      GVrms = G * KRMS * np.sqrt (Vd*Vd + Vq*Vq)
      grp_out = f_out.create_group (new_name)
      grp_out.create_dataset ('Step', data=step, compression='gzip')
      grp_out.create_dataset ('Ramp', data=ramp, compression='gzip')
      grp_out.create_dataset ('GIrms', data=GIrms, compression='gzip')
      grp_out.create_dataset ('GVrms', data=GVrms, compression='gzip')
      for tag in ['t', 'G', 'Ud', 'Uq', 'Vdc', 'Idc', 'Vd', 'Vq', 'Ctrl', 'Id', 'Iq', 'Rload']:
        grp_out.create_dataset (tag, data=grp_in[tag], compression='gzip')

  f_out.close()
  print ('Augmentation ends at {:s}'.format(new_name))

