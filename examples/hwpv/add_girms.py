# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: add_girms.py
""" Adds GIrms to existing data files.
"""

import numpy as np
import h5py
import math

cases = [
#  {"path":"d:/data/", "infile":"osg4_vdvq", "outfile":"osg4t_vdvq", "group":"case", "krms":0.001, "Id":"Id", "Iq":"Iq", "GIrms":"GIrms"},
#  {"path":"d:/data/", "infile":"ucf2", "outfile":"ucf2t", "group":"", "krms":math.sqrt(1.5), "Id":"Id", "Iq":"Iq", "GIrms":"GIrms"},
#  {"path":"d:/data/", "infile":"unb3", "outfile":"unb3t", "group":"case", "krms":0.001 * math.sqrt(1.5), "Id":"Idlo", "Iq":"Iqlo", "GIrms":"GIlo"},
   {"path":"d:/data/", "infile":"big3", "outfile":"big3t", "group":"case", "krms":0.001*math.sqrt(1.5), "Id":"Id", "Iq":"Iq", "GIrms":"GIrms"}
  ]

if __name__ == '__main__':
  for case in cases:
    input_path = case['path'] + case['infile'] + '.hdf5'
    output_path = case['path'] + case['outfile'] + '.hdf5'
    PREFIX = case['group']
    kRMS = case['krms']
    id_key = case['Id']
    iq_key = case['Iq']
    gi_key = case['GIrms']
    print ('Adding GIrms from {:s} to {:s}, prefix={:s}, k={:.4f}, in=[{:s},{:s}], out={:s}'.format (input_path, output_path, 
                                                                                                     PREFIX, kRMS, id_key, iq_key, gi_key))

    f_out = h5py.File (output_path, 'w')

    idx = 0
    n = -1

    with h5py.File(input_path, 'r') as f_in:
      ncases = len(f_in.items())
      for grp_name, grp_in in f_in.items():
        if idx % 10 == 0:
          print ('building {:d} of {:d} records'.format (idx, ncases))
        if n < 0:
          n = grp_in['t'].len()
        new_name = '{:s}{:d}'.format (PREFIX, idx)
        idx += 1
        G = np.zeros(n)
        Id = np.zeros(n)
        Iq = np.zeros(n)
        grp_in['G'].read_direct (G)
        grp_in[id_key].read_direct (Id)
        grp_in[iq_key].read_direct (Iq)
        GIrms = kRMS * G * np.sqrt (Id*Id + Iq*Iq)

        grp_out = f_out.create_group (new_name)
        grp_out.create_dataset (gi_key, data=GIrms, compression='gzip')
        for tag in grp_in:
          grp_out.create_dataset (tag, data=grp_in[tag], compression='gzip')

    f_out.close()

