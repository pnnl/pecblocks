# Copyright (C) 2024 Battelle Memorial Institute
import json
import os
import sys
import pandas as pd
import pecblocks.pv3_poly as pv3_model
import math
import numpy as np
import matplotlib.pyplot as plt

hdf5_filename = 'harness.hdf5'
KRMS = math.sqrt(1.5)

RGRID = 82.0

cases = [
  {
    'model': '../hwpv/bal3/bal3_fhf.json',
    'tmax': 10.0,
    'kGVrms': 0.001,
    'G':  [[-1.0, 1.0, 2.0, 100.0], 
           [0.0, 0.0, 950.0, 950.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 2.5, 3.0, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 1.2, 1.2]],
    'Uq': [[-1.0, 9.0, 9.010, 100.0], 
           [0.001, 0.001, -0.1, -0.1]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           # [8.0, 8.0, 5.0, 5.0]]
           [0.25, 0.25, 0.11, 0.11]]
  },
  {
    'model': '../hwpv/osg4/osg4_fhf.json',
    'tmax': 10.0,
    'kGVrms': 0.001,
    'G':  [[-1.0, 0.1, 1.1, 100.0], 
           [0.0, 0.0, 950.0, 950.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.0, 1.210, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 1.2, 1.2]],
    'Uq': [[-1.0, 9.0, 9.010, 100.0], 
           [0.001, 0.001, -0.1, -0.1]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           [3.0, 3.0, 2.0, 2.0]]
  },
  {
    'model': '../hwpv/ucf3/ucf3_fhf.json',
    'tmax': 2.0,
    'kGVrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1, 100.0], 
           [0.0, 0.0, 1000.0, 1000.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.0, 1.010, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 0.999995, 0.999995]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.001, 0.001, 0.001, 0.001]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           [90.0, 90.0, 65.0, 65.0]]
  },
  {
    'model': '../hwpv/ucf4n/ucf4n_fhf.json',
    'tmax': 2.0,
    'kGVrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1, 100.0], 
           [0.0, 0.0, 1000.0, 1000.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.0, 1.010, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 0.999995, 0.999995]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.001, 0.001, 0.001, 0.001]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           [90.0, 90.0, 65.0, 65.0]]
  },
  {
    'model': '../hwpv/ucf4nz/ucf4nz_fhf.json',
    'tmax': 2.0,
    'kGVrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1, 100.0], 
           [0.0, 0.0, 1000.0, 1000.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.0, 1.010, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 0.999995, 0.999995]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.001, 0.001, 0.001, 0.001]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           [90.0, 90.0, 65.0, 65.0]]
  }
]

def getInput(case, tag, t):
  xvals = case[tag][0]
  yvals = case[tag][1]
  return np.interp (t, xvals, yvals)

if __name__ == '__main__':

  print ('Usage: python harness.py [idx=0]')
  print ('Cases Available:')
  print ('Idx Model                                          Tmax')
  for i in range(len(cases)):
    print ('{:3d} {:45s} {:5.2f}'.format (i, cases[i]['model'], cases[i]['tmax']))

  case_idx = 0
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])

  case = cases[case_idx]
  fp = open (case['model'], 'r')
  cfg = json.load (fp)
  dt = cfg['t_step']
  tmax = case['tmax']
  kGVrms = case['kGVrms']
  fp.close()

  model = pv3_model.pv3 ()
  model.set_sim_config (cfg, model_only=False)
  model.start_simulation ()

  print ('Model inputs', model.COL_U, model.idx_in)
  print ('Model outputs', model.COL_Y, model.idx_out)
  print ('Case {:d}, dt={:.5f}, kGVrms={:.2f}'.format (case_idx, dt, kGVrms))

  t = 0.0
  nsteps = int (2.0 / dt) # for initialization of the model history terms
  Id = 0.0
  Iq = 0.0
  rows = []

#  print ('    Ts     Vd     Vq      G    GVrms     Md     Mq    Ctl    Vdc    Idc     Id     Iq')
  while t <= tmax:
    G = getInput(case, 'G', t)
    T = getInput(case, 'T', t)
    Md = getInput(case, 'Ud', t)
    Mq = getInput(case, 'Uq', t)
    Fc = getInput(case, 'Fc', t)
    Ctl = getInput(case, 'Ctl', t)
    Rg = getInput(case, 'Rg', t)
    Vd = Rg * Id
    Vq = Rg * Iq
    Vrms = KRMS * math.sqrt(Vd*Vd + Vq*Vq)
    GVrms = G * Vrms * kGVrms
 #   Vq *= 0.30

    if 'T' in model.COL_U and 'Fc' in model.COL_U:
      step_vals = [T, G, Fc, Md, Mq, Vd, Vq, GVrms, Ctl]
    elif 'Vq' in model.COL_U:
      step_vals = [G, Md, Mq, Vd, Vq, GVrms, Ctl]
    else:
      step_vals = [G, Md, Mq, Vd, GVrms, Ctl]
    Vdc, Idc, Id, Iq = model.step_simulation (step_vals, nsteps=nsteps)
    nsteps = 1

    Id = max(0.0, Id)

#    print ('{:6.3f} {:6.2f} {:6.2f} {:6.1f} {:8.1f} {:6.3f} {:6.3f} {:6.1f} {:6.2f} {:6.3f} {:6.3f} {:6.3f}'.format(t, 
#            Vd, Vq, G, GVrms, Md, Mq, Ctl, Vdc, Idc, Id, Iq))
    dict = {'t':t,'G':G,'T':T,'Md':Md,'Mq':Mq,'Fc':Fc,'Ctl':Ctl,'Rg':Rg,'Vd':Vd,'Vq':Vq,'GVrms':GVrms,'Vdc':Vdc,'Idc':Idc,'Id':Id,'Iq':Iq}
    rows.append (dict)
    t += dt

  print ('simulation done, writing output to', hdf5_filename)
  df = pd.DataFrame (rows)
  df.to_hdf (hdf5_filename, key='basecase', mode='w', complevel=9)

  df.plot(x='t', y=['G', 'T', 'Md', 'Mq', 'Fc', 
                    'Ctl', 'Rg', 'Vd', 'Vq', 'GVrms', 
                    'Vdc', 'Idc', 'Id', 'Iq'],
    title='Model {:s} Case {:d}'.format (case['model'], case_idx),
    layout=(3, 5), figsize=(15,8), subplots=True, grid=True)
  plt.show()
