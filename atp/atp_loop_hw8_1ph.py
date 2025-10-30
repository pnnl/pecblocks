# Copyright (C) 2018-2022 Battelle Memorial Institute
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

atp_dt = '1.000E-5'  # should be formatted to exactly fill 8 columns
atp_path = '.'
kdec = 200
method = 'slice'
#method = 'iir'
filtered = ['Vrms', 'Irms', 'Id', 'Iq', 'Vd', 'Vq', 'Vdc', 'Idc', 'I0', 'V0']
#method = None

# The inverter is rated 100 kW, 480 V
Vnom = 480.0
Pfull = 100.0e3

# for unbalanced parameter variations; start to a steady-state balanced condition
# 1T*5G*3F*3UD*3UQ*3P*6R = 2430 cases
Tvals = [35.0]
Gvals = [250.0, 450.0, 650.0, 825.0, 950.0]
FCvals = [57.5, 60.0, 62.5]
UDvals = [0.90, 0.95, 1.00]
UQvals = [-0.3, 0.001, 0.3]
Pvals = [0.85, 1.00, 1.15]
# unbalanced events
R2set = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]

def run_atp_case(atp_root, pl4_dest, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2, R2a, Tstep=2.0):
  atp_file = '{:s}.atp'.format (atp_root)
  prm_file = '{:s}.prm'.format (atp_root)
  lis_file = '{:s}.lis'.format (atp_root)
  pl4_file = '{:s}.pl4'.format (atp_root)
  fp = open (prm_file, mode='w')
  print ('$PARAMETER', file=fp)
  print ('__DELTAT   ={:s}'.format (atp_dt), file=fp)
  print ('____TMAX   =7.00', file=fp)
  print ('RLDA1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDB1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDC1_ ={:.3f}'.format (R1), file=fp)
  print ('RLDA2_ ={:.3f}'.format (R2a), file=fp)
  print ('RLDB2_ ={:.3f}'.format (R2), file=fp)
  print ('RLDC2_ ={:.3f}'.format (R2), file=fp)
  print ('G1________ ={:.4f}'.format (G1), file=fp)
  print ('DG________ ={:.4f}'.format (DG), file=fp)
  print ('TEMP0_____ ={:.4f}'.format (T1), file=fp)
  print ('DTEMP_____ ={:.4f}'.format (DT), file=fp)
  print ('FC0_______ ={:.4f}'.format (FC0), file=fp)
  print ('DFC_______ ={:.4f}'.format (DFC), file=fp)
  print ('UD0_______ ={:.6f}'.format (UD0), file=fp)
  print ('DUD_______ ={:.6f}'.format (DUD), file=fp)
  print ('UQ0_______ ={:.6f}'.format (UQ0), file=fp)
  print ('DUQ_______ ={:.6f}'.format (DUQ), file=fp)
  print ('BLANK END PARAMETER', file=fp)
  fp.close()
  cmdline = 'runtp ' + atp_file + ' >nul'
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

  # move the pl4 file
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'
  fp = open ('commands.script', mode='w')
  print ('file', atp_root, file=fp)
  print ('comtrade all', file=fp)
  print ('', file=fp)
  print ('stop', file=fp)
  fp.close()
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

def add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2, R2a):
  print ('{:3d} G[{:.1f},{:.1f}] T[{:.1f},{:.1f}] F[{:.1f},{:.1f}] UD[{:.3f},{:.3f}] UQ[{:.3f},{:.3f}] R[{:.3f},{:.3f}] R2a[{:.3f}]'.format(idx, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2, R2a))
#  return idx+1

  grp_name = 'group{:d}'.format(idx)
  if idx < 1:
    print ('writing', grp_name)
    mode = 'w'
  else:
    print ('appending', grp_name)
    mode = 'a'

  run_atp_case (atp_root, pl4_file,  G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2, R2a)
  channels = h5utils.load_atp_comtrade_channels (atp_root, filtered=filtered, k=kdec, method=method)
  h5utils.save_atp_channels (hdf5_file, grp_name, channels, mode=mode)

  return idx+1

if __name__ == '__main__':
  atp_root = sys.argv[1]
  pl4_path = sys.argv[2]
  hdf5_file = sys.argv[3]
  pl4_file = '{:s}/{:s}.pl4'.format (pl4_path, atp_root)
  print ('running {:s}, PL4 output to {:s}, hdf5 archive to {:s}'.format (atp_root, pl4_file, hdf5_file))
  ncases = len(Gvals)*len(Tvals)*len(FCvals)*len(Pvals)*len(UDvals)*len(UQvals)*len(R2set)
  print ('ncases = {:d}'.format(int(ncases)))
  idx = 0
  for T1 in Tvals:
    for G1 in Gvals:
      for Pmid in Pvals:
        P1 = Pfull*Pmid*(G1/1000.0)
        R1 = Vnom*Vnom/P1
        R2 = R1
        for FCmid in FCvals:
          for UDmid in UDvals:
            for UQmid in UQvals:
              for Rf in R2set:
                R2a = Rf * R2
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, 
                                G1=G1, 
                                DG=0.0001*G1, 
                                T1=T1, 
                                DT=0.0001*T1, 
                                FC0=FCmid, 
                                DFC=0.0001*FCmid, 
                                UD0=UDmid, 
                                DUD=0.0001*UDmid, 
                                UQ0=UQmid, 
                                DUQ=0.0001, 
                                R1=R1, 
                                R2=R2,
                                R2a=R2a)
#                quit()

