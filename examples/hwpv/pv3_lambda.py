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
model_path = './ucf7_config.json'

data_path = 'd:/data/ucf3/ucf9.hdf5'
model_path = './ucf9_config.json'

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

def sensitivity_analysis (model, idx, bPrint):
  nsteps = int (1.0 / model.t_step) # for initialization of the model history terms
  maxdIdVd = 0.0
  maxdIdVq = 0.0
  maxdIqVd = 0.0
  maxdIqVq = 0.0

  print ('   G0   Ud0   Uq0   Vd0    Vq0 Ctl   dIdVd   dIdVq   dIqVd   dIqVq')

  for G0 in [600.0, 800.0, 999.0]:
    for Ud0 in [0.8, 1.0, 1.2]:
      for Uq0 in [-0.5, 0.0, 0.5]:
        for Ctl in [0.0, 1.0]:
          for Vd0 in [170.0, 340.0]:
            for Vq0 in [-140.0, 0.0, 140.0]:
              # baseline
              model.start_simulation ()
              Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq0*Vq0)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd0, Vq0, GVrms, Ctl]
              Vdc0, Idc0, Id0, Iq0 = model.step_simulation (step_vals, nsteps=nsteps)

              # change Vd and GVrms
              model.start_simulation ()
              Vd1 = Vd0 + 1.0
              Vrms = KRMS * math.sqrt(Vd1*Vd1 + Vq0*Vq0)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd1, Vq0, GVrms, Ctl]
              Vdc1, Idc1, Id1, Iq1 = model.step_simulation (step_vals, nsteps=nsteps)

              # change Vq and GVrms
              model.start_simulation ()
              Vq1 = Vq0 + 1.0
              Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq1*Vq1)
              GVrms = G0 * Vrms
              step_vals = [G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
              Vdc2, Idc2, Id2, Iq2 = model.step_simulation (step_vals, nsteps=nsteps)

              # calculate the changes
              dIdVd = Id1 - Id0
              dIqVd = Iq1 - Iq0
              dIdVq = Id2 - Id0
              dIqVq = Iq2 - Iq0
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
  print ('                                     dIdVd   dIdVq   dIqVd   dIqVq')
  print ('{:34s} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format ('Maximum Magnitudes', maxdIdVd, maxdIdVq, maxdIqVd, maxdIqVq))


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

  if case_idx < 0:
    for idx in range(model.n_cases):
      sensitivity_analysis (model, idx, bPrint=True)
  else:
    sensitivity_analysis (model, case_idx, bPrint=True)

#  if case_idx < 0:
#    for idx in range(model.n_cases):
#      analyze_case (model, idx, bPrint=False)
#  else:
#    analyze_case (model, case_idx, bPrint=True)

#  print ('Rd range [{:.3f}-{:.3f}]'.format(minRd, maxRd))
#  print ('Rq range [{:.3f}-{:.3f}]'.format(minRq, maxRq))
#  print ('Event counts: G={:d} Ud={:d} Uq={:d} R={:d} Total={:d}'.format (nG, nUd, nUq, nRd, nG+nUd+nUq+nRd))
