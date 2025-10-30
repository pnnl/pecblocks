# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: pl4_to_hdf5.py
""" Convert ATP pl4 output to COMTRADE and then HDF5.

Public Functions:
    :main: does the work
"""

import math
import sys
import operator
import subprocess
import os
import shutil
import random
import h5utils
import numpy

atp_path = '.'

if __name__ == '__main__':
  atp_root = 'c:\\projects\\ucf_invcontrol\\atp_model\\pv1_osg' # sys.argv[1]
  pl4_file = '{:s}.pl4'.format (atp_root)
  hdf5_file = 'pv1_osg.hdf5'.format (atp_root)
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'

# fp = open ('commands.script', mode='w')
# print ('file', atp_root, file=fp)
# print ('comtrade all', file=fp)
# print ('', file=fp)
# print ('stop', file=fp)
# fp.close()
# pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
# pw0.wait()
# print ('created', atp_root, 'COMTRADE file')

  channels = h5utils.load_atp_comtrade_channels (atp_root, method=None, lab1=True)
  print ('created', atp_root, 'HDF5 file')
  h5utils.save_atp_channels (hdf5_file, 'base_case', channels, mode='w')

