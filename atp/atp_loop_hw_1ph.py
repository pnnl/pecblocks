# Copyright (C) 2018-2021 Battelle Memorial Institute
# file: AtpLoopFaults.py
""" Run all ATP training cases for HW models.

Called from ATP_Loop_HW.bat, driven by coded  parameter arrays.

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

atp_dt = '1.000E-4'  # should be formatted to exactly fill 8 columns
atp_path = '.'
kdec = 20
method = 'slice'
#method = 'iir'

# for parameter variations
Tvals = [0.0, 30.0]
Gvals = [200.0, 400.0, 600.0, 800.0, 1000.0]
Rvals = [2.304, 2.88, 3.84, 5.76, 11.52, 23.04, 230.4]
Vnom = 480.0
Pfull = 100.0e3
# The inverter is rated 100 kW, 480 V
# at 480 V, these Rvals draw
# Pvals = [100, 80, 60, 40, 20, 10, 1] kW

# Sliding-scale variation:
#   Use Tvals and Gvals to define steady-state operating weather points (10)
#   Nominal Load is then 100*(G/1000) kW
#   Pvals = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] * Pnom
#   Rvals = 480^2/Pvals
# The startups will be these 10*7=70
# From each startup condition:
#   Simulate SLGF with Rval=0.1 (1 case)
#   Step to the other Rvals on a single phase (6 cases)
#   There should be 70*7=490 cases; add to the balanced set for training
# Skipping the SLGF at 0.1 Ohms, there are 70*6=420 unbalanced cases

def run_atp_case(atp_root, pl4_dest, G1, G2, T1, T2, R1, R2a, Tstep=1.1):
#  tfault = random.uniform(0.15, 0.15 + 1/60)
#  vsrc = '{:.2f}'.format (atp_vpu * source_vbase)
  atp_file = '{:s}.atp'.format (atp_root)
  prm_file = '{:s}.prm'.format (atp_root)
  lis_file = '{:s}.lis'.format (atp_root)
  pl4_file = '{:s}.pl4'.format (atp_root)
  fp = open (prm_file, mode='w')
  print ('$PARAMETER', file=fp)
#  print ('_FLT_=\'' + bus.ljust(5) + '\'', file=fp)
  print ('__DELTAT   ={:s}'.format (atp_dt), file=fp)
  print ('____TMAX   =2.00', file=fp)
  print ('RLDA1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDB1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDC1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDA2_ ={:.3f}'.format (R2a), file=fp)
  print ('RLDB2_ ={:.3f}'.format (R1), file=fp)
  print ('RLDC2_ ={:.3f}'.format (R1), file=fp)
  print ('GSOL1_ ={:.1f}'.format (G1), file=fp)
  print ('GSOL2_ ={:.1f}'.format (G2), file=fp)
  print ('TEMP1_ ={:.1f}'.format (T1), file=fp)
  print ('TEMP2_ ={:.1f}'.format (T2), file=fp)
  print ('TSTEP_ ={:.3f}'.format (Tstep), file=fp)
  print ('BLANK END PARAMETER', file=fp)
  fp.close()
  cmdline = 'runtp ' + atp_file + ' >nul'
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

  # convert PL4 to COMTRADE in this directory
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'
  fp = open ('commands.script', mode='w')
  print ('file', atp_root, file=fp)
  print ('comtrade all', file=fp)
  print ('', file=fp)
  print ('stop', file=fp)
  fp.close()
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

def add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2a):
  grp_name = 'group{:d}'.format(idx)
  if idx < 1:
    print ('writing', grp_name)
    mode = 'w'
  else:
    print ('appending', grp_name)
    mode = 'a'

  run_atp_case (atp_root, pl4_file, G1, G2, T1, T2, R1, R2a)
  channels = h5utils.load_atp_comtrade_channels (atp_root, k=kdec, method=method)
  h5utils.save_atp_channels (hdf5_file, grp_name, channels, mode=mode)

  return idx+1

if __name__ == '__main__':
  atp_root = sys.argv[1]
  pl4_path = sys.argv[2]
  hdf5_file = 'looped_1ph.hdf5'
  pl4_file = '{:s}/{:s}.pl4'.format (pl4_path, atp_root)
  print ('running {:s}, PL4 output to {:s}, hdf5 archive to {:s}'.format (atp_root, pl4_file, hdf5_file))
  idx = 0
  for Ti in Tvals:
    for Gi in Gvals:
      Pnom = Pfull*(Gi/1000.0)
      Pvals=numpy.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])*Pnom
      Rvals=numpy.divide(numpy.full((7),Vnom*Vnom),Pvals)
      for Ri in Rvals:
        Rset = Rvals.tolist()
        Rset.remove (Ri)
#        Rset.append (0.1) # for SLGF
#        print ('{:3d}: Ti,Gi,Ri={:4.1f},{:6.1f},{:8.3f}'.format (idx, Ti, Gi, Ri))
#        print ('     Rset={:s}'.format (','.join(['{:.3f}'.format(item) for item in Rset])))
#        idx = idx + 1
        for Rf in Rset:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, Gi, Gi, Ti, Ti, Ri, Rf)

