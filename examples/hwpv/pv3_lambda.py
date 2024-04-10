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

KRMS = math.sqrt(1.5)

data_path = 'd:/data/ucf3/ucf7.hdf5'
model_path = './ucf7s_config.json'

#data_path = 'd:/data/ucf3/ucf9.hdf5'
#model_path = './ucf9_config.json'

#data_path = 'd:/data/ucf3/ucf9c.hdf5'
#model_path = './ucf10c_config.json'

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

def sensitivity_analysis (model, bPrint):
  maxdIdVd = 0.0
  maxdIdVq = 0.0
  maxdIqVd = 0.0
  maxdIqVq = 0.0

  if bPrint:
    print ('   G0   Ud0   Uq0   Vd0    Vq0 Ctl   dIdVd   dIdVq   dIqVd   dIqVq')
  model.start_simulation (bPrint=False)
  for G0 in [600.0, 800.0, 999.0]:
    for Ud0 in [0.8, 1.0, 1.2]:
      for Uq0 in [-0.5, 0.0, 0.5]:
        for Ctl in [0.0, 1.0]:
          for Vd0 in [170.0, 340.0]:
            for Vq0 in [-140.0, 0.0, 140.0]:
              # baseline
              Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq0*Vq0)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd0, Vq0, GVrms, Ctl]
              Vdc0, Idc0, Id0, Iq0 = model.steady_state_response (step_vals)

              # change Vd and GVrms
              Vd1 = Vd0 + 1.0
              Vrms = KRMS * math.sqrt(Vd1*Vd1 + Vq0*Vq0)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd1, Vq0, GVrms, Ctl]
              Vdc1, Idc1, Id1, Iq1 = model.steady_state_response (step_vals)

              # change Vq and GVrms
              Vq1 = Vq0 + 1.0
              Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq1*Vq1)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
              Vdc2, Idc2, Id2, Iq2 = model.steady_state_response (step_vals)

              # calculate the changes
              dIdVd = Id1 - Id0
              dIqVd = Iq1 - Iq0
              dIdVq = Id2 - Id0
              dIqVq = Iq2 - Iq0
              if bPrint:
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

if __name__ == '__main__':

  case_idx = 100 # 36 # 189
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])

  model_folder, config_file = os.path.split(model_path)
  model_root = config_file.rstrip('.json')
  model_root = model_root.rstrip('_config')
  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)

  model = pv3_model.pv3(training_config=model_path)
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization(filename=None, bSummary=True)
  model.initializeModelStructure()
  model.loadModelCoefficients()
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)
  #sens = sensitivity_analysis (model, bPrint=False)
  #print ('Maximum Sensitivity = {:.6f}'.format (sens))

  #print ('model.clamps', model.clamps)
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
