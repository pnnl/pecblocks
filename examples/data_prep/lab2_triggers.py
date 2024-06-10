import os
import sys
import math
import numpy as np
import glob

input_path = 'c:/data/outback/outbac*.csv'
output_path = 'c:/projects/ucf_invcontrol/lab/ltspice/'

# channel names
# Vdc = DCVoltage
# Idc = DCCurrent
# Vac = ACVoltage
# Iac = ACCurrent

if __name__ == '__main__':
  tbase = np.linspace (0.0, 0.6, 5000, endpoint=False)
  dt = tbase[1] - tbase[0]
  print ('tbase {:.8f} to {:.8f} at dt={:.8f}'.format (tbase[0], tbase[-1], dt))

  files = glob.glob (input_path)
  print ('Writing {:d} trigger files to {:s}'.format (len(files), output_path))
  idx = 1
  for fname in files:
    d = np.loadtxt (fname, delimiter=',', skiprows=1)
    trigger = -d[0,0]
    t = d[:,0] - d[0,0]
    idc = d[:,2]
    n = len(t)
    dtrec = (t[-1] - t[0]) / float(n-1.0)
    if dtrec > dt:
      idc = np.interp(tbase, t, idc.copy())
    fout = '{:s}trigger{:d}.txt'.format (output_path, idx)
    np.savetxt (fout, np.transpose([tbase,idc]), fmt='%.5f')
    idx += 1

