# Copyright (C) 2022-23 Battelle Memorial Institute
import json
import os
import sys
import h5py
import numpy as np
import math
import hwpv_evaluator as hwpv

METHOD = 'SBE'

cases = [
  {
    'model': 'balanced_fhf.json',
    'tmax': 5.0,
    'dt': 0.005,
    'G':  [[-1.0, 1.00,   2.00,   200.0], 
           [0.00, 0.00, 1000.0,  1000.0]],
    'T':  [[-1.0, 200.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 200.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 2.50, 2.51, 200.0],
           [0.00, 0.00, 1.00, 1.00]],
    'Ud': [[-1.00, 200.0], 
           [0.810, 0.810]],
    'Uq': [[-1.00, 200.0], 
           [0.392, 0.392]],
    'Rg': [[-1.0, 200.0], 
           [10.0, 10.0]]
  }
]

def getInput(case, tag, t):
  xvals = case[tag][0]
  yvals = case[tag][1]
  return np.interp (t, xvals, yvals)

def evaluation_loop(case, hdf5_filename):
  fp = open (case['model'], 'r')
  cfg = json.load (fp)
  fp.close()

  dt = case['dt']
  tmax = case['tmax']  
  n = int(tmax/dt) + 1

  mdl = hwpv.model ()
  mdl.set_sim_config (cfg)
  print ('Running {:s} model from {:s} using {:s}'.format (mdl.name, case['model'], METHOD))
  print ('  Input Channels: ', mdl.COL_U)
  print ('  Output Channels:', mdl.COL_Y)
  print ('  Training dt = {:.6f}s'.format (mdl.t_step))
  print ('  Running dt =  {:.6f}s to {:.4f}s for {:d} steps'.format (dt, tmax, n))
  # construct initial conditions, assuming Id=Iq=0, hence Vrms=GVrms=0
  T = getInput (case, 'T', 0.0)
  G = getInput (case, 'G', 0.0)
  Fc = getInput (case, 'Fc', 0.0)
  Ud = getInput (case, 'Ud', 0.0)
  Uq = getInput (case, 'Uq', 0.0)
  Ctl = getInput (case, 'Ctl', 0.0)
  if METHOD == 'Z':
    mdl.start_simulation_z (T=T, G=G, Fc=Fc, Ud=Ud, Uq=Uq, Vrms=0.0, GVrms=0.0, Ctl=Ctl)
  elif METHOD == 'SFE':
    mdl.start_simulation_sfe (T=T, G=G, Fc=Fc, Ud=Ud, Uq=Uq, Vrms=0.0, GVrms=0.0, Ctl=Ctl)
  elif METHOD == 'SBE':
    mdl.start_simulation_sbe (T=T, G=G, Fc=Fc, Ud=Ud, Uq=Uq, Vrms=0.0, GVrms=0.0, Ctl=Ctl, h=dt, log=True)
  Id = 0.0
  Iq = 0.0
  vals = np.zeros((n,13)) # t, 8 inputs, 4 outputs
  i = 0
  while i < n:
    t = dt * i
    T = getInput (case, 'T', t)
    G = getInput (case, 'G', t)
    Fc = getInput (case, 'Fc', t)
    Ud = getInput (case, 'Ud', t)
    Uq = getInput (case, 'Uq', t)
    Ctl = getInput (case, 'Ctl', t)
    R = getInput (case, 'Rg', t)
    Irms = math.sqrt(1.5) * math.sqrt(Id*Id + Iq*Iq)
    Vrms = Irms * R
    GVrms = G * Vrms
    if METHOD == 'Z':
      Vdc, Idc, Id, Iq = mdl.step_simulation_z (G=G, T=T, Ud=Ud, Uq=Uq, Fc=Fc, Vrms=Vrms, Ctl=Ctl, GVrms=GVrms)
    elif METHOD == 'SFE':
      Vdc, Idc, Id, Iq = mdl.step_simulation_sfe (G=G, T=T, Ud=Ud, Uq=Uq, Fc=Fc, Vrms=Vrms, Ctl=Ctl, GVrms=GVrms, h=dt)
    elif METHOD == 'SBE':
      Vdc, Idc, Id, Iq = mdl.step_simulation_sbe (G=G, T=T, Ud=Ud, Uq=Uq, Fc=Fc, Vrms=Vrms, Ctl=Ctl, GVrms=GVrms, h=dt)
    vals[i,:] = [t, G, T, Ud, Uq, Fc, Ctl, Vrms, GVrms, Vdc, Idc, Id, Iq]
    i += 1

  print ('Last Index', i, t, tmax)

  f = h5py.File (hdf5_filename, 'w')
  grp = f.create_group ('basecase')
  j = 0
  for key in ['t', 'G', 'T', 'Ud', 'Uq', 'Fc', 'Ctl', 'Vrms', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq']:
    grp.create_dataset (key, data=vals[:,j], compression='gzip')
    j += 1
  f.close()

if __name__ == '__main__':
  case_idx = 0
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
  case = cases[case_idx]
  evaluation_loop (case, 'hwpv_pi.hdf5')

