# Copyright (C) 2024 Battelle Memorial Institute
import json
import os
import sys
import pandas as pd
import pecblocks.pv3_poly as pv3_model
import numpy as np
import matplotlib.pyplot as plt

hdf5_filename = 'harness.hdf5'

cases = [
    {
      'model': '../hwpv/jan/ja_fhf.json',
      'tmax': 10.0,
      'Vdc':  [[-1.0, 0.1, 1.1, 100.0], 
             [600.0, 600.0, 600.0, 600.0]],
      'Fc': [[-1.0, 100.0],
             [60.0, 60.0]],
      'Ud': [[-1.0, 8.0, 8.010, 100.0], 
             [0.999995, 0.999995, 1.2, 1.2]],
      'Uq': [[-1.0, 9.0, 9.010, 100.0], 
             [0.001, 0.001, -0.1, -0.1]],
      'Rg': [[-1.0, 6.0, 6.010, 100.0], 
             [300.0, 300.0, 175.0, 175.0]]
    }
]

def getInput(case, tag, t):
  xvals = case[tag][0]
  yvals = case[tag][1]
  return np.interp (t, xvals, yvals)

if __name__ == '__main__':

  print ('Usage: python harness_sdi.py [idx=0]')
  print ('Cases Available:')
  print ('Idx Model                                     Tmax')
  for i in range(len(cases)):
    print ('{:3d} {:40s} {:5.2f}'.format (i, cases[i]['model'], cases[i]['tmax']))

  case_idx = 0
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])

  case = cases[case_idx]
  fp = open (case['model'], 'r')
  cfg = json.load (fp)
  dt = cfg['t_step']
  tmax = case['tmax']
  fp.close()

  model = pv3_model.pv3 ()
  model.set_sim_config (cfg, model_only=False)
  model.start_simulation ()

  print ('Model inputs', model.COL_U, model.idx_in)
  print ('Model outputs', model.COL_Y, model.idx_out)
  print ('Case {:d}, dt={:.5f}'.format (case_idx, dt))

  t = 0.0
  nsteps = int (2.0 / dt) # for initialization of the model history terms
  Id = 0.0
  Iq = 0.0
  rows = []

  while t <= tmax:
    Vdc = getInput(case, 'Vdc', t)
    Md = getInput(case, 'Ud', t)
    Mq = getInput(case, 'Uq', t)
    Fc = getInput(case, 'Fc', t)
    Rg = getInput(case, 'Rg', t)
    Vd = Rg * Id
    Vq = Rg * Iq

    step_vals = [Fc, Md, Mq, Vd, Vq, Vdc]
    Idc, Id, Iq = model.step_simulation (step_vals, nsteps=nsteps)
    nsteps = 1

    dict = {'t':t,'Md':Md,'Mq':Mq,'Fc':Fc,'Rg':Rg,'Vd':Vd,'Vq':Vq,'Vdc':Vdc,'Idc':Idc,'Id':Id,'Iq':Iq}
    rows.append (dict)
    t += dt

  print ('simulation done, writing output to', hdf5_filename)
  df = pd.DataFrame (rows)
  df.to_hdf (hdf5_filename, key='basecase', mode='w', complevel=9)

  df.plot(x='t', y=['Vdc', 'Md', 'Mq', 'Fc', 
                    'Rg', 'Vd', 'Vq', 
                    'Idc', 'Id', 'Iq'],
    title='Model {:s} Case {:d}'.format (case['model'], case_idx),
    layout=(3, 4), figsize=(15,8), subplots=True)
  plt.show()
