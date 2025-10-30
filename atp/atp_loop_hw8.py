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
filtered = ['Vrms', 'Irms', 'Id', 'Iq', 'Vd', 'Vq', 'Vdc', 'Idc']
#method = None

# The inverter is rated 100 kW, 480 V
Vnom = 480.0
Pfull = 100.0e3

# for parameter variations; start to the midpoint of each weather and electrical combination (2T*10G*1F*2UD*3UQ*3P=360)
Tvals = [15.0, 35.0]
Gvals = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
FCvals = [60.0]
UDvals = [0.95, 1.000]
UQvals = [-0.05, 0.001, 0.05]
Pvals = [0.95, 1.00, 1.05]
# from each startup position, create disturbances to values below (1T, 4G, 20P, 10F, 10Ud, 20Uq)
# total would be 360*65 = 23400 cases
DFset = [-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
DUDset = [-0.20, -0.18, -0.16, -0.14, 0.12, -0.10, -0.08, -0.06, -0.04, -0.02]
DUQset = [-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
          0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
Pset = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
        1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20]

# for a reduced set; 4 weather conditions * 11 steps (2G, 1T, 2P, 2F, 2Ud, 2Uq) = 44 cases
#Gvals = [500.0, 1000.0]
#DFset = [-5.0, 5.0]
#DUDset = [-0.1, 0.1]
#DUQset = [-0.1, 0.1]
#Pset = [0.85, 1.15]

def g_disturbance_set (G1):
  if G1 >= 999.0:
    return [900.0, 800.0, 700.0, 600.0]
  elif G1 >= 899.0:
    return [1000.0, 950.0, 850.0, 800.0]
  elif G1 >= 799.0:
    return [1000.0, 900.0, 700.0, 600.0]
  elif G1 >= 699.0:
    return [900.0, 800.0, 600.0, 500.0]
  elif G1 >= 599.0:
    return [800.0, 700.0, 500.0, 400.0]
  elif G1 >= 499.0:
    return [700.0, 600.0, 400.0, 300.0]
  elif G1 >= 399.0:
    return [600.0, 500.0, 200.0, 100.0]
  elif G1 >= 299.0:
    return [500.0, 400.0, 200.0, 100.0]
  elif G1 >= 199.0:
    return [300.0, 250.0, 150.0, 100.0]
  else:
    return [50.0, 150.0, 200.0, 250.0]

def run_atp_case(atp_root, pl4_dest, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2, Tstep=2.0):
  atp_file = '{:s}.atp'.format (atp_root)
  prm_file = '{:s}.prm'.format (atp_root)
  lis_file = '{:s}.lis'.format (atp_root)
  pl4_file = '{:s}.pl4'.format (atp_root)
  fp = open (prm_file, mode='w')
  print ('$PARAMETER', file=fp)
  print ('__DELTAT   ={:s}'.format (atp_dt), file=fp)
  print ('____TMAX   =7.00', file=fp)
  print ('RLOD1_ ={:.3f}'.format (R1), file=fp)
  print ('RLOD2_ ={:.3f}'.format (R2), file=fp)
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
#  print ('moving {:s} to {:s}'.format (pl4_file, pl4_dest))
#  shutil.move (pl4_file, pl4_dest)
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'
  # commands.script already exists ... 
  # file GFM_v8
  # comtrade all
  # 
  # stop
  # 
  fp = open ('commands.script', mode='w')
  print ('file', atp_root, file=fp)
  print ('comtrade all', file=fp)
  print ('', file=fp)
  print ('stop', file=fp)
  fp.close()
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

def add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2):
#  if idx < 21290:
#    return idx+1
  print ('{:3d} G[{:.1f},{:.1f}] T[{:.1f},{:.4f}] F[{:.1f},{:.4f}] UD[{:.4f},{:.4f}] UQ[{:.4f},{:.4f}] R[{:.3f},{:.3f}]'.format(idx, G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2))
#  return idx+1

  grp_name = 'group{:d}'.format(idx)
  if idx < 1:
    print ('writing', grp_name)
    mode = 'w'
  else:
    print ('appending', grp_name)
    mode = 'a'

  run_atp_case (atp_root, pl4_file,  G1, DG, T1, DT, FC0, DFC, UD0, DUD, UQ0, DUQ, R1, R2)
  channels = h5utils.load_atp_comtrade_channels (atp_root, filtered=filtered, k=kdec, method=method)
  h5utils.save_atp_channels (hdf5_file, grp_name, channels, mode=mode)

#  quit()

  return idx+1

def make_test_set (atp_root, pl4_path, hdf5_file):
  pl4_file = '{:s}/{:s}.pl4'.format (pl4_path, atp_root)
  print ('Test Set: running {:s}, PL4 output to {:s}, hdf5 archive to {:s}'.format (atp_root, pl4_file, hdf5_file))
  T1 = 25.0
  G1s = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 925.0, 975.0]
  F1 = 60.0
  UD1 = 1.000
  UQ1 = 0.001
  Pmid = 1.0
  # from each startup position, create disturbances to values below (6G, 6P, 6F, 6Ud, 6Uq)
  # total would be 10*30 = 300 cases
  DFs = [-4.5, -2.5, -0.5, 0.5, 2.5, 4.5]
  DUDs = [-0.19, -0.15, -0.13, -0.09, -0.05, -0.03]
  DUQs = [-0.43, -0.23, -0.13, 0.13, 0.23, 0.43]
  P2s = [0.83, 0.91, 0.97, 1.03, 1.09, 1.17]
  idx = 0
  for G1 in G1s:
    P1 = Pfull*Pmid*(G1/1000.0)
    R1 = Vnom*Vnom/P1
    R2 = R1
    Gset = G1s.copy()
    Gset.remove (G1)
    del Gset[::3]
#    print (G1, R1, Gset)
    for G2 in Gset:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, DG=G2-G1, T1=T1, DT=0.0001, 
                              FC0=F1, DFC=0.0001, UD0=UD1, DUD=0.0001, UQ0=UQ1, DUQ=0.0001, R1=R1, R2=R2)
    for DF in DFs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, DG=-0.0001, T1=T1, DT=0.0001, 
                              FC0=F1, DFC=DF, UD0=UD1, DUD=0.0001, UQ0=UQ1, DUQ=0.0001, R1=R1, R2=R2)
    for DUD in DUDs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, DG=-0.0001, T1=T1, DT=0.0001, 
                              FC0=F1, DFC=0.0001, UD0=UD1, DUD=DUD, UQ0=UQ1, DUQ=0.0001, R1=R1, R2=R2)
    for DUQ in DUQs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, DG=-0.0001, T1=T1, DT=0.0001, 
                              FC0=F1, DFC=0.0001, UD0=UD1, DUD=0.0001, UQ0=UQ1, DUQ=DUQ, R1=R1, R2=R2)
    for P2 in P2s:
      R2 = R1 / P2
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, DG=-0.0001, T1=T1, DT=0.0001, 
                              FC0=F1, DFC=0.0001, UD0=UD1, DUD=0.0001, UQ0=UQ1, DUQ=0.0001, R1=R1, R2=R2)
  print ('  next idx={:d}'.format(idx))

if __name__ == '__main__':
  atp_root = sys.argv[1]
  pl4_path = sys.argv[2]
  hdf5_file = sys.argv[3]
  if len(sys.argv) > 4:
    if sys.argv[4] == 'testset':
      make_test_set (atp_root, pl4_path, hdf5_file)
      quit()

  pl4_file = '{:s}/{:s}.pl4'.format (pl4_path, atp_root)
  print ('running {:s}, PL4 output to {:s}, hdf5 archive to {:s}'.format (atp_root, pl4_file, hdf5_file))
  ncases = (len(Gvals)*len(Tvals)*len(FCvals)*len(Pvals)*len(UDvals)*len(UQvals)) * (len(DFset)+len(DUDset)+len(DUQset)+len(Pset)+1+4)
  print ('ncases = {:d}'.format(int(ncases)))
  idx = 0
  for T1 in Tvals:
    Tset = Tvals.copy()
    Tset.remove (T1)
    for G1 in Gvals:
      Gset = g_disturbance_set (G1)
      for Ppu in Pvals:
        P1 = Pfull*Ppu*(G1/1000.0)
        R1 = Vnom*Vnom/P1
        R2 = R1
        for UD in UDvals:
          for UQ in UQvals:
            for FC in FCvals:
              for Gf in Gset:
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file,
                                G1=G1,
                                DG=Gf-G1,
                                T1=T1,
                                DT=0.0001*T1,
                                FC0=FC,
                                DFC=0.0001*FC,
                                UD0=UD,
                                DUD=0.0001*UD,
                                UQ0=UQ,
                                DUQ=0.0001,
                                R1=R1,
                                R2=R2)
              for Tf in Tset:
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file,
                                G1=G1,
                                DG=-0.0001*G1,
                                T1=T1,
                                DT=Tf-T1,
                                FC0=FC,
                                DFC=0.0001*FC,
                                UD0=UD,
                                DUD=0.0001*UD,
                                UQ0=UQ,
                                DUQ=0.0001,
                                R1=R1,
                                R2=R2)
              for DF in DFset:
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file,
                                G1=G1,
                                DG=-0.0001*G1,
                                T1=T1,
                                DT=0.0001*T1,
                                FC0=FC,
                                DFC=DF,
                                UD0=UD,
                                DUD=0.0001*UD,
                                UQ0=UQ,
                                DUQ=0.0001,
                                R1=R1,
                                R2=R2)
              for DUD in DUDset:
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file,
                                G1=G1,
                                DG=-0.0001*G1,
                                T1=T1,
                                DT=0.0001*T1,
                                FC0=FC,
                                DFC=0.0001*FC,
                                UD0=UD,
                                DUD=DUD,
                                UQ0=UQ,
                                DUQ=0.0001,
                                R1=R1,
                                R2=R2)
              for DUQ in DUQset:
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file,
                                G1=G1,
                                DG=-0.0001*G1,
                                T1=T1,
                                DT=0.0001*T1,
                                FC0=FC,
                                DFC=0.0001*FC,
                                UD0=UD,
                                DUD=0.0001*UD,
                                UQ0=UQ,
                                DUQ=DUQ,
                                R1=R1,
                                R2=R2)
              for P2 in Pset:
                R2 = R1 / P2
                idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, 
                                G1=G1, 
                                DG=-0.0001*G1, 
                                T1=T1, 
                                DT=0.0001*T1, 
                                FC0=FC, 
                                DFC=0.0001*FC, 
                                UD0=UD, 
                                DUD=0.0001*UD, 
                                UQ0=UQ, 
                                DUQ=0.0001, 
                                R1=R1, 
                                R2=R2)

