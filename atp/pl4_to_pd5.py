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
import h5utils
import pandas as pd
import numpy as np
from comtrade import Comtrade
#import matplotlib.pyplot as plt

atp_root = 'TACS_HWPV_Test'
atp_root = 'c:\\projects\\ucf_invcontrol\\atp_model\\pv1_osg' # sys.argv[1]
atp_path = '.'

def make_pd_key (prefix, tok1, tok2 = None):
  key = '{:s}:{:s}'.format (prefix, tok1.lstrip().rstrip().replace('_', ''))
  if tok2 is not None:
    key = '{:s}:{:s}'.format (key, tok2.lstrip().rstrip().replace('_', ''))
  return key

if __name__ == '__main__':
  if len(sys.argv) > 1:
    atp_root = sys.argv[1]
  pl4_file = '{:s}.pl4'.format (atp_root)
  hdf5_file = '{:s}.hdf5'.format (atp_root)
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'

  fp = open ('commands.script', mode='w')
  print ('file', atp_root, file=fp)
  print ('comtrade all', file=fp)
  print ('', file=fp)
  print ('stop', file=fp)
  fp.close()
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()
  print ('created', atp_root, 'COMTRADE file')

  chan = {}
  rec = Comtrade ()
  rec.load (atp_root + '.cfg', atp_root + '.dat')
  t = np.array(rec.time)
  t = np.linspace (t[0], t[-1], len(t)) # COMTRADE does not have sub-microsecond time steps
  for i in range(rec.analog_count):
    lbl = rec.analog_channel_ids[i]
    tok1 = lbl[0:6]
    tok2 = lbl[7:13]
    tok3 = lbl[14:]
#    print ('"{:s}" "{:s}" "{:s}" "{:s}"'.format (lbl, tok1, tok2, tok3))
    key = None
    if tok1 == 'TACS  ':
      key = make_pd_key ('T', tok2)
    elif tok1 == 'MODELS':
      key = make_pd_key ('M', tok2)
    elif tok3 == 'V-branch':
      key = make_pd_key ('V', tok1, tok2)
    elif tok3 == '  V-node':
      key = make_pd_key ('V', tok1)
    elif tok3 == 'I-branch':
      key = make_pd_key ('I', tok1, tok2)
    else:
      print ('Unrecognized COMTRADE Channel {:d} "{:s}"'.format (i, lbl))
    if key is not None:
      chan[key] = np.array (rec.analog[i])

  df = pd.DataFrame(data=chan, index=t)
  df.info()
  df.to_hdf(atp_root + '.hdf5', key='Base', mode='w', complevel=4)

#  ax = df.plot (title='TACS HWPV Test', subplots=True)
#  plt.show()

