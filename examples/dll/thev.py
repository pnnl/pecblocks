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
    'model': '../hwpv/ucf4tsiir/ucf4tsiir_fhf.json',
    'tmax': 6.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 1.801, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf4ts2nd/ucf4ts2nd_fhf.json',
    'tmax': 6.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 1.801, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf10t1siir/ucf10t1siir_fhf.json',
    'tmax': 4.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 2.510, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf10t1s/ucf10t1s_fhf.json',
    'tmax': 4.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 2.510, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf10t1/ucf10t1_fhf.json',
    'tmax': 4.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 2.510, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf10t2/ucf10t2_fhf.json',
    'tmax': 4.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1,    2.0,   3.0, 100.0], 
           [0.0, 0.0, 900.0, 900.0, 850.0, 850.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 2.510, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [1.05, 1.05, 1.05, 1.05]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.051, 0.051, 0.051, 0.051]],
    'Rg': [[-1.0, 3.0,   3.010, 100.0], 
           [100.0, 100.0, RGRID, RGRID]]
  },
  {
    'model': '../hwpv/ucf10thev/ucf10thev_fhf.json',
    'tmax': 4.0,
    'kGIrms': 1.0,
    'G':  [[-1.0, 0.1, 1.1, 100.0], 
           [0.0, 0.0, 950.0, 950.0]],
    'T':  [[-1.0, 100.0],
           [35.0, 35.0]],
    'Fc': [[-1.0, 100.0],
           [60.0, 60.0]],
    'Ctl':[[-1.0, 1.8, 2.510, 100.0],
           [0.0, 0.0, 1.0, 1.0]],
    'Ud': [[-1.0, 8.0, 8.010, 100.0], 
           [0.999995, 0.999995, 0.999995, 0.999995]],
    'Uq': [[-1.0, 8.0, 8.010, 100.0], 
           [0.001, 0.001, 0.001, 0.001]],
    'Rg': [[-1.0, 6.0, 6.010, 100.0], 
           [95.0, 95.0, 65.0, 65.0]]
  }
]

def getInput(case, tag, t):
  xvals = case[tag][0]
  yvals = case[tag][1]
  return np.interp (t, xvals, yvals)

if __name__ == '__main__':

  print ('Usage: python thev.py [idx=0]')
  print ('Cases Available:')
  print ('Idx Model                                     Tmax')
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
  kGIrms = case['kGIrms']
  fp.close()

  model = pv3_model.pv3 ()
  model.set_sim_config (cfg, model_only=False)
  model.start_simulation ()

  print ('Model inputs', model.COL_U, model.idx_in)
  print ('Model outputs', model.COL_Y, model.idx_out)
  print ('Case {:d}, dt={:.5f}, kGIrms={:.2f}'.format (case_idx, dt, kGIrms))

  t = 0.0
  nsteps = int (2.0 / dt) # for initialization of the model history terms
  Vd = 0.0
  Vq = 0.0
  rows = []
  if 'GIrms' not in model.COL_U:
    kGIrms = 0.0

  while t <= tmax:
    G = getInput(case, 'G', t)
    T = getInput(case, 'T', t)
    Md = getInput(case, 'Ud', t)
    Mq = getInput(case, 'Uq', t)
    Fc = getInput(case, 'Fc', t)
    Ctl = getInput(case, 'Ctl', t)
    Rg = getInput(case, 'Rg', t)
    Id = Vd / Rg
    Iq = Vq / Rg
    Irms = KRMS * math.sqrt(Id*Id + Iq*Iq)
    GIrms = G * Irms * kGIrms

    Vrms = KRMS * math.sqrt(Vd*Vd + Vq*Vq)
    GVrms = G * Vrms * kGIrms

    if 'GIrms' in model.COL_U:
      if 'T' in model.COL_U and 'Fc' in model.COL_U:
        step_vals = [T, G, Fc, Md, Mq, Id, Iq, GIrms, Ctl]
      elif 'Iq' in model.COL_U:
        step_vals = [G, Md, Mq, Id, Iq, GIrms, Ctl]
      else:
        step_vals = [G, Md, Mq, Id, GIrms, Ctl]
    elif 'GVrms' in model.COL_U:
      if 'T' in model.COL_U and 'Fc' in model.COL_U:
        step_vals = [T, G, Fc, Md, Mq, Id, Iq, GVrms, Ctl]
      elif 'Iq' in model.COL_U:
        step_vals = [G, Md, Mq, Id, Iq, GVrms, Ctl]
      else:
        step_vals = [G, Md, Mq, Id, GVrms, Ctl]
    else:
      if 'T' in model.COL_U and 'Fc' in model.COL_U:
        step_vals = [T, G, Fc, Md, Mq, Id, Iq, Ctl]
      elif 'Iq' in model.COL_U:
        step_vals = [G, Md, Mq, Id, Iq, Ctl]
      else:
        step_vals = [G, Md, Mq, Id, Ctl]

    Vdc, Idc, Vd, Vq = model.step_simulation (step_vals, nsteps=nsteps)
    nsteps = 1

#    Vd = max(0.0, Id)

    dict = {'t':t,'G':G,'T':T,'Md':Md,'Mq':Mq,'Fc':Fc,'Ctl':Ctl,'Rg':Rg,'Vd':Vd,'Vq':Vq,'GIrms':GIrms,'Vdc':Vdc,'Idc':Idc,'Id':Id,'Iq':Iq}
    rows.append (dict)
    t += dt

  print ('simulation done, writing output to', hdf5_filename)
  df = pd.DataFrame (rows)
  df.to_hdf (hdf5_filename, key='basecase', mode='w', complevel=9)

  df.plot(x='t', y=['G', 'T', 'Md', 'Mq', 'Fc', 
                    'Ctl', 'Rg', 'Id', 'Iq', 'GIrms', 
                    'Vdc', 'Idc', 'Vd', 'Vq'],
    title='Model {:s} Case {:d}'.format (case['model'], case_idx),
    layout=(3, 5), figsize=(15,8), subplots=True, grid=True)
  plt.show()
