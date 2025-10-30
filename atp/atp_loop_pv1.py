# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: atp_loop_pv1.py
""" Run all ATP training cases for HW models.

Called from ATP_Loop_PV1.bat, driven by coded  parameter arrays.

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

bTesting = False

atp_path = '.'
kdec = 400
#method = 'slice'
method = 'iir'
method=None
method='butter'
filtered = ['Vrms', 'Irms', 'Id', 'Iq', 'Vd', 'Vq', 'Vdc', 'Idc']

# for parameter variations
# at 1000 ms, ramp to each combo of Tvals and Gvals, with midpoints Fvals, Uvals, Ppus (2*10=20)
Tvals = [15.0, 35.0]
#Gvals = [150.0, 300.0, 500.0, 700.0, 850.0, 1000.0]
Gvals = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
# at 4000 ms, change to one of the other Fvals, UDvals, UQvals, Ppus, Gvals (4+3+4+4+4=19), 380 total cases
Fset = [55.0, 58.0, 62.0, 65.0]
UDset = [0.8, 0.9, 1.1]
UQset = [-0.4, -0.2, 0.2, 0.4]
Pset = [0.8, 0.9, 1.1, 1.2]
Fmid = 60.0
UDmid = 1.0
UQmid = 0.001
Pmid = 1.0
# nominal inverter output rating
Vnom = 240.0
Pfull = 13.5e3

#def g_disturbance_set (G1):
#  if G1 >= 999.0:
#    return [900.0, 800.0]
#  elif G1 >= 749.0:
#    return [900.0, 600.0]
#  elif G1 >= 499.0:
#    return [600.0, 400.0]
#  else:
#    return [300.0, 200.0]

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

# given G, the nominal power is Pout=13500*(G/1000) [W]
# nominal resistance is then Rnom=240^2/Pout [Ohms]
# for the given Gvals, Rnom=[42.677, 10.667, 6.0952, 4.2667]
#   to perturb Pout, Rnom /= 1.2 or Rnom /= 0.8

def run_atp_case(atp_root, pl4_dest, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2):
  atp_file = '{:s}.atp'.format (atp_root)
  prm_file = '{:s}.prm'.format (atp_root)
  lis_file = '{:s}.lis'.format (atp_root)
  pl4_file = '{:s}.pl4'.format (atp_root)
  fp = open (prm_file, mode='w')
  print ('$PARAMETER', file=fp)
  print ('G1________={:.5f}'.format (G1), file=fp)
  print ('DG________={:.5f}'.format (G2-G1), file=fp)
  print ('TEMP0_____={:.6f}'.format (T1), file=fp)
  print ('DTEMP_____={:.6f}'.format (T2-T1), file=fp)
  print ('UD0_______={:.6f}'.format (UD1), file=fp)
  print ('DUD_______={:.6f}'.format (UD2-UD1), file=fp)
  print ('UQ0_______={:.6f}'.format (UQ1), file=fp)
  print ('DUQ_______={:.6f}'.format (UQ2-UQ1), file=fp)
  print ('FC0_______={:.6f}'.format (F1), file=fp)
  print ('DFC_______={:.6f}'.format (F2-F1), file=fp)
  print ('RLOD1_={:.3f}'.format (R1), file=fp)
  print ('RLOD2_={:.3f}'.format (R2), file=fp)
  print ('BLANK END PARAMETER', file=fp)
  fp.close()
  cmdline = 'runtp ' + atp_file + ' >nul'
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

  # move the pl4 file
#  print ('moving {:s} to {:s}'.format (pl4_file, pl4_dest))
#  shutil.move (pl4_file, pl4_dest)
  cmdline = 'c:\\atp\\gtppl32\\gtppl32 @@commands.script > nul'
  fp = open ('commands.script', mode='w')
  print ('file', atp_root, file=fp)
  print ('comtrade all', file=fp)
  print ('', file=fp)
  print ('stop', file=fp)
  fp.close()
  pw0 = subprocess.Popen (cmdline, cwd=atp_path, shell=True)
  pw0.wait()

def add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2):
#  if idx < 115:
#    return idx+1
  grp_name = 'group{:d}'.format(idx)
  print ('{:3d} {:s} G=[{:.1f},{:.1f}] T=[{:.1f},{:.1f}] R=[{:.3f},{:.3f}] F=[{:.2f},{:.2f}] UD=[{:.3f},{:.3f}] UQ=[{:.3f},{:.3f}]'.format(idx, grp_name,
    G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2))
#  return idx+1
  if idx < 1:
    print ('  writing', grp_name)
    mode = 'w'
  else:
    print ('  appending', grp_name)
    mode = 'a'

  run_atp_case (atp_root, pl4_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)
  channels = h5utils.load_atp_comtrade_channels (atp_root, filtered=filtered, k=kdec, method=method, pv1=True)
  h5utils.save_atp_channels (hdf5_file, grp_name, channels, mode=mode)

  # save the prm, lis, and pl4 files
#  shutil.copyfile ('pv1_osg.pl4', 'case{:d}.pl4'.format(idx))
#  shutil.copyfile ('pv1_osg.lis', 'case{:d}.lis'.format(idx))
#  shutil.copyfile ('pv1_osg.prm', 'case{:d}.prm'.format(idx))

  return idx+1

def make_test_set (atp_root, pl4_path, hdf5_file):
  pl4_file = '{:s}/{:s}.pl4'.format (pl4_path, atp_root)
  print ('Test Set: running {:s}, PL4 output to {:s}, hdf5 archive to {:s}'.format (atp_root, pl4_file, hdf5_file))
  T1 = 25.0
  T2 = 25.1
  G1s = [400.0, 900.0]
  F1 = 60.0
  UD1 = 1.000
  UQ1 = 0.001
  Pmid = 1.0
  # from each startup position, create disturbances to values below (1G, 2P, 2F, 2Ud, 2Uq)
  # total would be 2*9 = 18 cases
  DFs = [-2.5, 2.5]
  DUDs = [-0.05, 0.05]
  DUQs = [-0.30, 0.30]
  P2s = [0.9, 1.1]
  idx = 0
  for G1 in G1s:
    P1 = Pfull*Pmid*(G1/1000.0)
    R1 = Vnom*Vnom/P1
    R2 = R1
    Gset = G1s.copy()
    Gset.remove (G1)
    for G2 in Gset:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, G2=G2, T1=T1, T2=T2, R1=R1, R2=R2, 
                              F1=F1, F2=F1+0.01, UD1=UD1, UD2=UD1+0.001, UQ1=UQ1, UQ2=UQ1+0.001)
    G2 = G1 - 0.1
    for DF in DFs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, G2=G2, T1=T1, T2=T2, R1=R1, R2=R2, 
                              F1=F1, F2=F1+DF,     UD1=UD1, UD2=UD1+0.001, UQ1=UQ1, UQ2=UQ1+0.001)
    for DUD in DUDs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, G2=G2, T1=T1, T2=T2, R1=R1, R2=R2, 
                              F1=F1, F2=F1+0.01, UD1=UD1, UD2=UD1+DUD,   UQ1=UQ1, UQ2=UQ1+0.001)
    for DUQ in DUQs:
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, G2=G2, T1=T1, T2=T2, R1=R1, R2=R2, 
                              F1=F1, F2=F1+0.01, UD1=UD1, UD2=UD1+0.001, UQ1=UQ1, UQ2=UQ1+DUQ)
    for P2 in P2s:
      R2 = R1 / P2
      idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1=G1, G2=G2, T1=T1, T2=T2, R1=R1, R2=R2, 
                              F1=F1, F2=F1+0.01, UD1=UD1, UD2=UD1+0.001, UQ1=UQ1, UQ2=UQ1+0.001)

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
  idx = 0
  for T1 in Tvals:
    T2 = T1 + 0.0001
    F1 = Fmid
    UD1 = UDmid
    UQ1 = UQmid
    for G1 in Gvals:
#      Gset = Gvals.copy()
#      Gset.remove (G1)
      Gset = g_disturbance_set (G1)
      P1 = Pfull*Pmid*(G1/1000.0)
      R1 = Vnom*Vnom / P1
      print ('Startup: G1={:.1f} T1={:.1f} R1={:.3f} F1={:.1f} UD1={:.3f} UQ1={:.3f}'.format(G1, T1, R1, F1, UD1, UQ1))
      F2 = F1 + 0.000001
      UD2 = UD1 + 0.000001
      UQ2 = UQ1 + 0.000001
      R2 = R1
      for G2 in Gset:  # vary G from this starting point
        if bTesting:
          print (' idx={:d} G2={:.1f} R2={:.3f} F2={:.2f} UD2={:.3f} UQ2={:.3f}'.format(idx, G2, R2, F2, UD2, UQ2))
          idx += 1
        else:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)
      G2 = G1 + 0.0001  # reset to starting condition; vary P from this starting point
      for P2 in Pset:
        R2 = R1 / P2
        if bTesting:
          print (' idx={:d} G2={:.1f} R2={:.3f} F2={:.2f} UD2={:.3f} UQ2={:.3f}'.format(idx, G2, R2, F2, UD2, UQ2))
          idx += 1
        else:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)
      R2 = R1  # reset to starting condition, vary F from this starting point
      for F2 in Fset:
        if bTesting:
          print (' idx={:d} G2={:.1f} R2={:.3f} F2={:.2f} UD2={:.3f} UQ2={:.3f}'.format(idx, G2, R2, F2, UD2, UQ2))
          idx += 1
        else:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)
      F2 = F1 + 0.000001 # reset to starting condition, vary U from this starting point
      for UD2 in UDset:
        if bTesting:
          print (' idx={:d} G2={:.1f} R2={:.3f} F2={:.2f} UD2={:.3f} UQ2={:.3f}'.format(idx, G2, R2, F2, UD2, UQ2))
          idx += 1
        else:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)
      UD2 = UD1 + 0.000001
      for UQ2 in UQset:
        if bTesting:
          print (' idx={:d} G2={:.1f} R2={:.3f} F2={:.2f} UD2={:.3f} UQ2={:.3f}'.format(idx, G2, R2, F2, UD2, UQ2))
          idx += 1
        else:
          idx = add_training_set (idx, atp_root, pl4_file, hdf5_file, G1, G2, T1, T2, R1, R2, F1, F2, UD1, UD2, UQ1, UQ2)

