# Copyright (C) 2022-24 Battelle Memorial Institute
import json
import os
import sys
import h5py
import numpy as np
import math
import hwpv_evaluator as hwpv

# choices are SBE (preferred, s-domain backward Euler), 
# SFE (s-domain forward Euler), and Z (discrete domain)
METHOD = 'Z' # 'SBE'

# interpolation tables for each case are keyed as:
#   'Rg' is a fixed key for external load resistance
#   other keys will be searched for a matching key in the model's COL_U
cases = [
  {
    'model': 'bal3_fhf.json',
    'group': 'bal3',
    'tmax': 8.0,
    'dt': 0.002,
    'krms': math.sqrt(1.5),
    'kGVrms': 0.001,
    'G':  [[-1.0, 1.00,   2.00,   200.0], 
           [0.00, 0.00, 1000.0,  1000.0]],
    'T':  [[-1.0, 200.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 200.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 2.50, 2.51, 200.0],
           [0.00, 0.00, 1.00, 1.00]],
    'Md': [[-1.00, 200.0], 
           [1.000, 1.000]],
    'Mq': [[-1.00, 200.0], 
           [0.000, 0.000]],
    'Rg': [[-1.0, 5.000, 5.001, 200.0], 
           [2.3, 2.3, 3.0, 3.0]]
  },
  {
    'model': '../hwpv/bal3/bal3_fhf.json',
    'group': 'bal3',
    'tmax': 5.0,
    'dt': 0.002, # .001,
    'krms': math.sqrt(1.5),
    'kGVrms': 0.001,
    'G':  [[-1.0, 1.00,   2.00,   200.0], 
           [0.00, 0.00, 1000.0,  1000.0]],
    'T':  [[-1.0, 200.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 200.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 2.50, 2.51, 200.0],
           [0.00, 0.00, 1.00, 1.00]],
    'Md': [[-1.00, 200.0], 
           [1.000, 1.000]],
    'Mq': [[-1.00, 200.0], 
           [0.000, 0.000]],
    'Rg': [[-1.0, 200.0], 
           [0.2, 0.2]]
  },
  {
    'model': 'balanced_fhf.json',
    'group': 'vrms',
    'tmax': 5.0,
    'dt': 0.005,
    'krms': math.sqrt(1.5),
    'kGVrms': 1.0,
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
  },
  {
    'model': '../hwpv/ucf4n/ucf4n_fhf.json',
    'group': 'ucf4n',
    'tmax': 5.0,
    'dt': 0.001,
    'krms': math.sqrt(1.5),
    'kGVrms': 1.0,
    'G':  [[-1.0, 1.00,   2.00,   200.0], 
           [0.00, 0.00, 1000.0,  1000.0]],
    'T':  [[-1.0, 200.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 200.0],
           [60.0, 60.0]],
    'Ramp':[[-1.0, 2.50, 2.51, 200.0],
           [0.00, 0.00, 1.00, 1.00]],
    'Ud': [[-1.00, 200.0], 
           [0.810, 0.810]],
    'Uq': [[-1.00, 200.0], 
           [0.392, 0.392]],
    'Rg': [[-1.0, 200.0], 
           [65.0, 65.0]]
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

  # find the input variable indices for COL_U
  tag_indices = {}
  for key in case:
    if key in mdl.COL_U:
      tag_indices[key] = mdl.COL_U.index(key)
  print ('Direct Input Indices to COL_U', tag_indices)
  if 'G' in tag_indices:
    G_idx = tag_indices['G']
  else:
    G_idx = -1
    print ('** WARNING: solar irradiance (G) should be a direct input')
  calc_indices = {}
  for key in mdl.COL_U:
    if key not in tag_indices:
      calc_indices[key] = mdl.COL_U.index(key)
  print ('Calculated Input Indices to COL_U', calc_indices)

  # construct initial conditions, assuming Id=Iq=0, hence Vrms=GVrms=0
  inputs = np.zeros (mdl.nin)
  for tag, j in tag_indices.items():
    inputs[j] = getInput (case, tag, 0.0)
  if METHOD == 'Z':
    mdl.start_simulation_z (inputs)
  elif METHOD == 'SFE':
    mdl.start_simulation_sfe (inputs)
  elif METHOD == 'SBE':
    mdl.start_simulation_sbe (inputs, h=dt, log=False)
  Id = 0.0
  Iq = 0.0
  vals = np.zeros((n, 1 + mdl.nin + mdl.nout))
  i = 0
  while i < n:
    t = dt * i
    for tag, j in tag_indices.items():
      inputs[j] = getInput (case, tag, t)
    R = getInput (case, 'Rg', t)
    if 'Vrms' in calc_indices:
      Irms = case['krms'] * math.sqrt(Id*Id + Iq*Iq)
      Vrms = Irms * R
      inputs[calc_indices['Vrms']] = Vrms
    elif 'Vd' in calc_indices and 'Vq' in calc_indices:
      Vd = Id * R
      Vq = Iq * R
      Vrms = case['krms'] * math.sqrt(Vd*Vd + Vq*Vq)
      inputs[calc_indices['Vd']] = Vd
      inputs[calc_indices['Vq']] = Vq

    if 'GVrms' in calc_indices:
      GVrms = inputs[G_idx] * Vrms * case['kGVrms']
      inputs[calc_indices['GVrms']] = GVrms

    if METHOD == 'Z':
      outputs = mdl.step_simulation_z (inputs)
    elif METHOD == 'SFE':
      outputs = mdl.step_simulation_sfe (inputs, h=dt)
    elif METHOD == 'SBE':
      outputs = mdl.step_simulation_sbe (inputs, h=dt)
    Id = outputs[2]
    Iq = outputs[3]
    vals[i, 0] = t
    vals[i, 1:mdl.nin+1] = inputs
    vals[i, mdl.nin+1:] = outputs
    i += 1

  print ('Last Index', i, t, tmax)

  f = h5py.File (hdf5_filename, 'w')
  grp = f.create_group (case['group'])
  j = 0
  keys = ['t']
  for key in mdl.COL_U: # substitute Ud and Uq for Md and Mq
    if key == 'Md':
      keys.append ('Ud')
    elif key == 'Mq':
      keys.append ('Uq')
    else:
      keys.append(key)
  for key in mdl.COL_Y:
    keys.append(key)
  print ('saving {:s} as group {:s} to {:s}'.format (str(keys), case['group'], hdf5_filename))
  for key in keys:
    grp.create_dataset (key, data=vals[:,j], compression='gzip')
    j += 1
  f.close()

if __name__ == '__main__':
  case_idx = 0
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
  case = cases[case_idx]
  evaluation_loop (case, 'hwpv_pi.hdf5')

