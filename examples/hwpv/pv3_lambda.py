# copyright 2021-2024 Battelle Memorial Institute
# tabulates estimates of Req and lambda
#  arg1: case number to plot 1..ncases (default 100)
#
# example: python pv3_lambda.py 200

import pandas as pd
import numpy as np
import os
import sys
import pecblocks.pv3_poly as pv3_model
import math
import json
import torch

KRMS = math.sqrt(1.5)

minRd = 1.0e9
maxRd = 0.0
minRq = 1.0e9
maxRq = 0.0

nRd = 0
nG = 0
nUd = 0
nUq = 0

def analyze_case(model, idx, bPrint=False):
  global minRd, maxRd, minRq, maxRq
  global nRd, nG, nT, nFc, nUd, nUq
  i1 = 180 # 990
  i2 = 380 # 2499

  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=100)

  jId = model.COL_Y.index ('Id')
  jIq = model.COL_Y.index ('Iq')
  jVd = model.COL_U.index ('Vd')
  jVq = model.COL_U.index ('Vq')

  jG = model.COL_U.index ('G')
  jUd = model.COL_U.index ('Ud')
  jUq = model.COL_U.index ('Uq')

  dG = (u[i2,jG] - u[i1,jG]) * model.normfacs['G']['scale']
  dUd = (u[i2,jUd] - u[i1,jUd]) * model.normfacs['Ud']['scale']
  dUq = (u[i2,jUq] - u[i1,jUq]) * model.normfacs['Uq']['scale']

  Rd1 = abs ((u[i1,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']) / (y_true[i1,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  Rd2 = abs ((u[i2,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']) / (y_true[i2,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  dRd = Rd2 - Rd1
  dRflag = '**'
  if abs(dG) > 40.0:
    nG += 1
    dRflag = ''
  elif abs(dUd) > 0.01:
    nUd += 1
    dRflag = ''
  elif abs(dUq) > 0.01:
    nUq += 1
    dRflag = ''
  if dRflag == '**':
    nRd += 1

  if bPrint:
    print ('Case {:d}: dG={:.2f} dUd={:.2f} dUq={:.2f} Rd={:.2f} dRd={:.2f} {:s}'.format(idx, dG, dUd, dUq, Rd1, dRd, dRflag))

  for i in [i1, i2]:
    t = model.t[i]
    Id = y_true[i,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']
    Iq = y_true[i,jIq] * model.normfacs['Iq']['scale'] + model.normfacs['Iq']['offset']
    Vd = u[i,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']
    Vq = u[i,jVq] * model.normfacs['Vq']['scale'] + model.normfacs['Vq']['offset']
    Rd = abs(Vd/Id)
    if Rd < minRd:
      minRd = Rd
    if Rd > maxRd:
      maxRd = Rd
    Rq = abs(Vq/Iq)
    if Rq < minRq:
      minRq = Rq
    if Rq > maxRq:
      maxRq = Rq
#    print ('  t={:6.3f} Id={:.3f} Iq={:6.3f} Vod={:8.3f} Voq={:8.3f} Rd={:8.2f} Rq={:8.2f}'.format (t, Id, Iq, Vod, Voq, Rd, Rq))

def clamp_loss (model, bPrint):
  if not hasattr (model, 'clamp'):
    return 0.0
  total, cases = model.clampingErrors (bByCase=True)
  total = total.detach().numpy()

  if bPrint:
    out_size = len(model.COL_Y)
    colstr = ','.join('{:s}'.format(col) for col in model.COL_Y)
    print ('Clamping Loss by Case and Output')
    print ('Idx,{:s}'.format(colstr))
    for i in range(len(cases)):
      valstr = ','.join('{:.6f}'.format(cases[i][j]) for j in range(out_size))
      print ('{:d},{:s}'.format(i, valstr))
    print ('Total Clamping Error Summary')
    for j in range(out_size):
      print ('{:4s} Loss={:8.6f}'.format (model.COL_Y[j], total[j]))
  return np.sum(total)

def build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl):
  if math.isnan(T0):
    if math.isnan(F0):
      return [G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
    else:
      return [G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
  elif math.isnan(F0):
    return [T0, G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
  return [T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]

def sensitivity_analysis (model, bPrint, bLog = False, bAutoRange = False):
  maxdIdVd = 0.0
  maxdIdVq = 0.0
  maxdIqVd = 0.0
  maxdIqVq = 0.0
  delta = 0.01

  T_range = [math.nan]
  Fc_range = [math.nan]
  if bAutoRange:
    print ('Autoranging:')
    print ('Column       Min       Max      Mean     Range')
    idx = 0
    for c in model.COL_U + model.COL_Y:
      fac = model.normfacs[c]
      dmean = fac['offset']
      if 'max' in fac and 'min' in fac:
        dmax = fac['max']
        dmin = fac['min']
      else:
        dmax = model.de_normalize (np.max (model.data_train[:,:,idx]), fac)
        dmin = model.de_normalize (np.min (model.data_train[:,:,idx]), fac)
      drange = dmax - dmin
      if abs(drange) <= 0.0:
        drange = 1.0
      print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, dmin, dmax, dmean, drange))
      if c in ['G']:
        G_range = np.linspace (max (dmin, 50.0), dmax, 7) # 11)
      elif c in ['Ud', 'Md']:
        Ud_range = np.linspace (dmin, dmax, 5) # 5)
      elif c in ['Uq', 'Mq']:
        Uq_range = np.linspace (dmin, dmax, 5) # 5)
      elif c in ['Ctl', 'Ctrl']:
        Ctl_range = np.linspace (0.0, 1.0, 2)
      elif c in ['Vd']:
        Vd_range = np.linspace (dmin, dmax, 5) # 11)
      elif c in ['Vq']:
        Vq_range = np.linspace (dmin, dmax, 5) # 11)
      elif c in ['T']:
        T_range = np.linspace (dmin, dmax, 2)
      elif c in ['Fc']:
        Fc_range = np.linspace (dmin, dmax, 3) # 5)
      idx +=     1
  else:
    G_range = np.linspace (100.0, 1000.0, 10)
    Ud_range = np.linspace (0.8, 1.2, 5)
    Uq_range = np.linspace (-0.5, 0.5, 5)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Vd_range = np.linspace (170.0, 340.0, 5)
    Vq_range = np.linspace (-140.0, 140.0, 5)
  ncases = len(G_range) * len(Ud_range) * len(Uq_range) * len(Ctl_range) * len(Vd_range) * len(Vq_range) * len(T_range) * len(Fc_range)
  print ('Using {:d} sensitivity cases'.format (ncases))
  print (' T_range', T_range)
  print (' Fc_range', Fc_range)
  print (' G_range', G_range)
  print (' Ud_range', Ud_range)
  print (' Uq_range', Uq_range)
  print (' Ctl_range', Ctl_range)
  print (' Vd_range', Vd_range)
  print (' Vq_range', Vq_range)

  if bLog:
    print ('   G0   Ud0   Uq0   Vd0    Vq0 Ctl   dIdVd   dIdVq   dIqVd   dIqVq')
  model.start_simulation (bPrint=False)
  for T0 in T_range:
    for F0 in Fc_range:
      for G0 in G_range:
        print ('checking G={:.2f}'.format(G0), 'T=', T0, 'Fc=', F0)
        for Ud0 in Ud_range:
          for Uq0 in Uq_range:
            for Ctl in Ctl_range:
              for Vd0 in Vd_range:
                for Vq0 in Vq_range:
                  # baseline
                  Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq0*Vq0)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq0, GVrms, Ctl)
                  Vdc0, Idc0, Id0, Iq0 = model.steady_state_response (step_vals)

                  # change Vd and GVrms
                  Vd1 = Vd0 + delta
                  Vrms = KRMS * math.sqrt(Vd1*Vd1 + Vq0*Vq0)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd1, Vq0, GVrms, Ctl)
                  Vdc1, Idc1, Id1, Iq1 = model.steady_state_response (step_vals)

                  # change Vq and GVrms
                  Vq1 = Vq0 + delta
                  Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq1*Vq1)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl)
                  Vdc2, Idc2, Id2, Iq2 = model.steady_state_response (step_vals)

                  # calculate the changes
                  dIdVd = (Id1 - Id0) / delta
                  dIqVd = (Iq1 - Iq0) / delta
                  dIdVq = (Id2 - Id0) / delta
                  dIqVq = (Iq2 - Iq0) / delta
                  if bLog:
                    print ('{:6.1f} {:4.2f} {:5.2f} {:5.1f} {:6.1f} {:3.1f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format (G0, Ud0, Uq0, Vd0, Vq0, Ctl,
                                                                                                                     dIdVd, dIdVq, dIqVd, dIqVq))
                  # track the global maxima
                  dIdVd = abs(dIdVd)
                  if dIdVd > maxdIdVd:
                    maxdIdVd = dIdVd

                  dIdVq = abs(dIdVq)
                  if dIdVq > maxdIdVq:
                    maxdIdVq = dIdVq

                  dIqVd = abs(dIqVd)
                  if dIqVd > maxdIqVd:
                    maxdIqVd = dIqVd

                  dIqVq = abs(dIqVq)
                  if dIqVq > maxdIqVq:
                    maxdIqVq = dIqVq
  if bPrint:
    print ('                                     dIdVd   dIdVq   dIqVd   dIqVq')
    print ('{:34s} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format ('Maximum Magnitudes', maxdIdVd, maxdIdVq, maxdIqVd, maxdIqVq))
  return max(maxdIdVd, maxdIdVq, maxdIqVd, maxdIqVq)

def find_channel_indices (targets, available):
  n = len(targets)
  idx = []
  for i in range(n):
    idx.append (available.index(targets[i]))
  return n, idx

counter = 0

def build_baselines (bases, step_vals, cfg, keys, indices, lens, level):
  global counter
  counter += 1

  key = keys[level]
  ary = cfg['sets'][key]
  idx = cfg['idx_set'][key]

  if level+1 == len(keys): # add basecases at the lowest level
    for i in range(lens[level]):
      step_vals[idx] = ary[i]
      bases.append (step_vals.copy())
  else: # propagate this new value down to lower levels
    step_vals[idx] = ary[indices[level]]

  if level+1 < len(keys):
    level += 1
    build_baselines (bases, step_vals, cfg, keys, indices, lens, level)
  else:
    level -= 1
    while level >= 0:
      if indices[level]+1 >= lens[level]:
        level -= 1
      else:
        indices[level] += 1
        indices[level+1:] = 0
        build_baselines (bases, step_vals, cfg, keys, indices, lens, level)

def get_gvrms (G, Vd, Vq, k):
  return G * k * math.sqrt(Vd*Vd + Vq*Vq)

def model_sensitivity (model, bPrint):
  cfg = model.sensitivity
  cfg['n_in'], cfg['idx_in'] = find_channel_indices (cfg['inputs'], model.COL_U)
  cfg['n_out'], cfg['idx_out'] = find_channel_indices (cfg['outputs'], model.COL_Y)
  cfg['idx_set'] = {}
  for key in cfg['sets']:
    cfg['idx_set'][key] = model.COL_U.index(key)
  print ('inputs', cfg['inputs'], cfg['idx_in'])
  print ('outputs', cfg['outputs'], cfg['idx_out'])
  print ('sets', cfg['idx_set'])
  cfg['idx_g_rms'] = model.COL_U.index (cfg['GVrms']['G'])
  cfg['idx_vd_rms'] = model.COL_U.index (cfg['GVrms']['Vd'])
  cfg['idx_vq_rms'] = model.COL_U.index (cfg['GVrms']['Vq'])
  cfg['idx_gvrms'] = model.COL_U.index ('GVrms')
  krms = cfg['GVrms']['k']
  print ('GVrms', cfg['idx_g_rms'], cfg['idx_vd_rms'], cfg['idx_vq_rms'], krms, cfg['idx_gvrms'])

  max_sens = np.zeros ((cfg['n_out'], cfg['n_in']))
  sens = np.zeros ((cfg['n_out'], cfg['n_in']))
  step_vals = np.zeros (len(model.COL_U))

  bases = []
  keys = list(cfg['sets'])
  indices = np.zeros(len(keys), dtype=int)
  lens = np.zeros(len(keys), dtype=int)
  for i in range(len(keys)):
    lens[i] = len(cfg['sets'][keys[i]])
  build_baselines (bases, step_vals, cfg, keys, indices, lens, 0)
  print (len(bases), 'base cases', counter, 'function calls')

  model.start_simulation (bPrint=False)
  for vals in bases:
    Vd0 = vals[cfg['idx_vd_rms']]
    Vq0 = vals[cfg['idx_vq_rms']]
    Vd1 = Vd0 + 1.0
    Vq1 = Vq0 + 1.0

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd0, Vq0, krms)
    _, _, Id0, Iq0 = model.steady_state_response (vals.copy())
    #print (vals, Id0, Iq0)

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd1, Vq0, krms)
    vals[cfg['idx_vd_rms']] = Vd1
    vals[cfg['idx_vq_rms']] = Vq0
    _, _, Id1, Iq1 = model.steady_state_response (vals.copy())
    #print (vals, Id1, Iq1)

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd0, Vq1, krms)
    vals[cfg['idx_vd_rms']] = Vd0
    vals[cfg['idx_vq_rms']] = Vq1
    _, _, Id2, Iq2 = model.steady_state_response (vals.copy())
    #print (vals, Id2, Iq2)
    #prevent aliasing the base cases
    vals[cfg['idx_vq_rms']] = Vq0

    sens[0][0] = abs(Id1 - Id0)
    sens[1][0] = abs(Iq1 - Iq0)
    sens[0][1] = abs(Id2 - Id0)
    sens[1][1] = abs(Iq2 - Iq0)
    for i in range(2):
      for j in range(2):
        if sens[i][j] > max_sens[i][j]:
          max_sens[i][j] = sens[i][j]

  if bPrint:
    print (max_sens)
  return np.max(max_sens)

def thevenin_sensitivity_analysis (model, bPrint, bLog = False, bReducedSet = False):
  maxdVdId = 0.0
  maxdVdIq = 0.0
  maxdVqId = 0.0
  maxdVqIq = 0.0
  delta = 0.01

  if bReducedSet:
    # 72 cases for faster training
    G_range = np.linspace (300.0, 1000.0, 3)
    Ud_range = np.linspace (1.0, 1.1, 2)
    Uq_range = np.linspace (-0.5, 0.5, 2)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Id_range = np.linspace (3.0, 6.0, 3)
    Iq_range = np.linspace (-2.7, -2.7, 1)
  else:
    # 60500 cases for more accuracy
    G_range = np.linspace (100.0, 1000.0, 10)
    Ud_range = np.linspace (0.8, 1.2, 5)
    Uq_range = np.linspace (-0.5, 0.5, 5)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Id_range = np.linspace (-0.1, 6.0, 11)
    Iq_range = np.linspace (-2.7, 2.4, 11)

  ncases = len(G_range) * len(Ud_range) * len(Uq_range) * len(Ctl_range) * len(Id_range) * len(Iq_range)
  bGVrms = 'GVrms' in model.COL_U
  bGIrms = 'GIrms' in model.COL_U
  worst_dVdId = None
  worst_dVdIq = None
  worst_dVqId = None
  worst_dVqIq = None
  print ('Using {:d} sensitivity cases with dI={:.3f}, GVrms {:s}, GIrms {:s}'.format (ncases, delta, str (bGVrms), str (bGIrms)))
  if bLog:
    print ('   G0   Ud0   Uq0   Id0    Iq0 Ctl   dVdId   dVdIq   dVqId   dVqIq')
  model.start_simulation (bPrint=False)
  for G0 in G_range:
    print (' checking G0={:.2f}'.format (G0))
    for Ud0 in Ud_range:
      for Uq0 in Uq_range:
        # estimate GVrms, ignoring any feedback effects from Id and Iq
        Vdx = np.interp (Ud0, [0.8, 1.21], [170.0, 350.0])
        Vqx = np.interp (Uq0, [-0.5, 0.5], [-154.0, 144.0])
        Vrms = KRMS * math.sqrt(Vdx*Vdx + Vqx*Vqx)
        GVrms = G0 * Vrms
        for Ctl in Ctl_range:
          for Id0 in Id_range:
            for Iq0 in Iq_range:
              # baseline
              Irms = KRMS * math.sqrt(Id0*Id0 + Iq0*Iq0)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, Ctl]
              Vdc0, Idc0, Vd0, Vq0 = model.steady_state_response (step_vals)

              # change Id
              Id1 = Id0 + delta
              Irms = KRMS * math.sqrt(Id1*Id1 + Iq0*Iq0)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, Ctl]
              Vdc1, Idc1, Vd1, Vq1 = model.steady_state_response (step_vals)

              # change Iq
              Iq1 = Iq0 + delta
              Irms = KRMS * math.sqrt(Id0*Id0 + Iq1*Iq1)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, Ctl]
              Vdc2, Idc2, Vd2, Vq2 = model.steady_state_response (step_vals)

              # calculate the changes
              dVdId = (Vd1 - Vd0) / delta
              dVqId = (Vq1 - Vq0) / delta
              dVdIq = (Vd2 - Vd0) / delta
              dVqIq = (Vq2 - Vq0) / delta
              if bLog:
                print ('{:6.1f} {:4.2f} {:5.2f} {:5.1f} {:6.1f} {:3.1f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format (G0, Ud0, Uq0, Id0, Iq0, Ctl,
                                                                                                                 dVdId, dVdIq, dVqId, dVqIq))
              # track the global maxima
              dVdId = abs(dVdId)
              if dVdId > maxdVdId:
                maxdVdId = dVdId
                worst_dVdId = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVdIq = abs(dVdIq)
              if dVdIq > maxdVdIq:
                maxdVdIq = dVdIq
                worst_dVdIq = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVqId = abs(dVqId)
              if dVqId > maxdVqId:
                maxdVqId = dVqId
                worst_dVqId = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVqIq = abs(dVqIq)
              if dVqIq > maxdVqIq:
                maxdVqIq = dVqIq
                worst_dVqIq = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

  if bPrint:
    print ('                                     dVdId   dVdIq   dVqId   dVqIq')
    print ('{:34s} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format ('Maximum Magnitudes', maxdVdId, maxdVdIq, maxdVqId, maxdVqIq))
    print ('worst case dVdId [G, Ud, Uq, Id, Iq]:', worst_dVdId)
    print ('worst case dVdIq [G, Ud, Uq, Id, Iq]:', worst_dVdIq)
    print ('worst case dVqId [G, Ud, Uq, Id, Iq]:', worst_dVqId)
    print ('worst case dVqIq [G, Ud, Uq, Id, Iq]:', worst_dVqIq)
  return max(maxdVdId, maxdVdIq, maxdVqId, maxdVqIq)

if __name__ == '__main__':
  if len(sys.argv) > 1:
    config_file = sys.argv[1]
    fp = open (config_file, 'r')
    cfg = json.load (fp)
    fp.close()
    data_path = cfg['data_path']
    model_folder = cfg['model_folder']
    model_root = cfg['model_root']
  else:
    print ('Usage: python pv3_lambda.py config.json')
    quit()

  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)

  case_idx = 100 # 36 # 189
  if len(sys.argv) > 2:
    case_idx = int(sys.argv[2])

  model = pv3_model.pv3(training_config=config_file)
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization(filename=None, bSummary=True)
  model.initializeModelStructure()
  model.loadModelCoefficients()
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)
  if 'Vd' in model.COL_U and 'Vq' in model.COL_U and 'Id' in model.COL_Y and 'Iq' in model.COL_Y:
    sens = sensitivity_analysis (model, bPrint=True, bAutoRange=True)
    print ('Maximum Norton Sensitivity = {:.6f}'.format (sens))
    clamp = clamp_loss (model, bPrint=True)
    print ('Total Norton Clamping Loss = {:.6f}'.format (clamp))
  elif 'Id' in model.COL_U and 'Iq' in model.COL_U and 'Vd' in model.COL_Y and 'Vq' in model.COL_Y:
    sens = thevenin_sensitivity_analysis (model, bPrint=True)
    print ('Maximum Thevenin Sensitivity = {:.6f}'.format (sens))
  else:
    print ('No Thevenin or Norton columns found: skipping sensitivity analysis')
    #sens = sensitivity_analysis (model, bPrint=True, bLog=False, bAutoRange=True)
    #print ('Maximum Auto Sensitivity = {:.6f}'.format (sens))

  quit()

  #print ('model.clamps', model.clamps)
  if 'sensitivity' in cfg:
    print ('model.sensitivity', model.sensitivity)
    sens = model_sensitivity (model, bPrint=True)
    print ('Maximum Sensitivity = {:.6f}'.format (sens))
    sens_loss = max (sens - model.sensitivity['limit'], 0.0)
    print ('Sensitivity Loss = {:.6f}'.format (sens_loss))
    rmse, mae, case_rmse, case_mae = model.trainingErrors(True)
    loss_rmse = model.n_cases * rmse * rmse
    print ('RMSE =', rmse, 'Loss RMSE =', loss_rmse)
    print ('Total Loss =', model.trainingLosses())

#  if case_idx < 0:
#    for idx in range(model.n_cases):
#      analyze_case (model, idx, bPrint=False)
#  else:
#    analyze_case (model, case_idx, bPrint=True)

#  print ('Rd range [{:.3f}-{:.3f}]'.format(minRd, maxRd))
#  print ('Rq range [{:.3f}-{:.3f}]'.format(minRq, maxRq))
#  print ('Event counts: G={:d} Ud={:d} Uq={:d} R={:d} Total={:d}'.format (nG, nUd, nUq, nRd, nG+nUd+nUq+nRd))
