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

data_path = 'd:/data/ucf2.hdf5'
model_path = './ucf2ac/ucf2ac_config.json'

minRd = 1.0e9
maxRd = 0.0
minRq = 1.0e9
maxRq = 0.0

nRd = 0
nG = 0
nT = 0
nFc = 0
nMd1 = 0
nMq1 = 0

def analyze_case(model, idx):
  global minRd, maxRd, minRq, maxRq, nRd, nG, nT, nFc, nMd1, nMq1
  i1 = 990
  i2 = 2499

  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=500)

  jId = model.COL_Y.index ('Id')
  jIq = model.COL_Y.index ('Iq')
  jVod = model.COL_U.index ('Vod')
  jVoq = model.COL_U.index ('Voq')

  jG = model.COL_U.index ('G')
  jT = model.COL_U.index ('T')
  jFc = model.COL_U.index ('Fc')
  jMd1 = model.COL_U.index ('Md1')
  jMq1 = model.COL_U.index ('Mq1')

  dG = (u[i2,jG] - u[i1,jG]) * model.normfacs['G']['scale']
  dT = (u[i2,jT] - u[i1,jT]) * model.normfacs['T']['scale']
  dFc = (u[i2,jFc] - u[i1,jFc]) * model.normfacs['Fc']['scale']
  dMd1 = (u[i2,jMd1] - u[i1,jMd1]) * model.normfacs['Md1']['scale']
  dMq1 = (u[i2,jMq1] - u[i1,jMq1]) * model.normfacs['Mq1']['scale']

  Rd1 = abs ((u[i1,jVod] * model.normfacs['Vod']['scale'] + model.normfacs['Vod']['offset']) / (y_true[i1,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  Rd2 = abs ((u[i2,jVod] * model.normfacs['Vod']['scale'] + model.normfacs['Vod']['offset']) / (y_true[i2,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  dRd = Rd2 - Rd1
  dRflag = '**'
  if abs(dG) > 10.0:
    nG += 1
    dRflag = ''
  if abs(dT) > 1.0:
    nT += 1
    dRflag = ''
  if abs(dFc) > 0.5:
    nFc += 1
    dRflag = ''
  if abs(dMd1) > 0.01:
    nMd1 += 1
    dRflag = ''
  if abs(dMq1) > 0.01:
    nMq1 += 1
    dRflag = ''
  if dRflag == '**':
    nRd += 1

  print ('Case {:d}: dG={:.2f} dT={:.2f} dFc={:.2f} dMd1={:.2f} dMq1={:.2f} Rd={:.2f} dRd={:.2f} {:s}'.format(idx, dG, dT, dFc, dMd1, dMq1, Rd1, dRd, dRflag))

  for i in [i1, i2]:
    t = model.t[i]
    Id = y_true[i,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']
    Iq = y_true[i,jIq] * model.normfacs['Iq']['scale'] + model.normfacs['Iq']['offset']
    Vod = u[i,jVod] * model.normfacs['Vod']['scale'] + model.normfacs['Vod']['offset']
    Voq = u[i,jVoq] * model.normfacs['Voq']['scale'] + model.normfacs['Voq']['offset']
    Rd = abs(Vod/Id)
    if Rd < minRd:
      minRd = Rd
    if Rd > maxRd:
      maxRd = Rd
    Rq = abs(Voq/Iq)
    if Rq < minRq:
      minRq = Rq
    if Rq > maxRq:
      maxRq = Rq
#    print ('  t={:6.3f} Id={:.3f} Iq={:6.3f} Vod={:8.3f} Voq={:8.3f} Rd={:8.2f} Rq={:8.2f}'.format (t, Id, Iq, Vod, Voq, Rd, Rq))

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
  model.loadAndApplyNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)

  if case_idx < 0:
    for idx in range(model.n_cases):
      analyze_case (model, idx)
  else:
    analyze_case (model, case_idx)

  print ('Rd range [{:.3f}-{:.3f}]'.format(minRd, maxRd))
  print ('Rq range [{:.3f}-{:.3f}]'.format(minRq, maxRq))
  print ('Event counts: G={:d} T={:d} Fc={:d} Md1={:d} Mq1={:d} R={:d}'.format (nG, nT, nFc, nMd1, nMq1, nRd))
