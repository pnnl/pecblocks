# copyright 2021-2024 Battelle Memorial Institute
# tabulates estimates of Req and sigma
#  arg1: case number to plot 1..ncases (default 100)
#
# example for Mac, from ../dev:  python3 ../hwpv/pv3_sigma.py ucf4ts2nd_config.json

import os
import sys
import pecblocks.pv3_poly as pv3_model
import pecblocks.pv3_functions as pv3_fn
import json

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
    print ('Usage: python pv3_sigma.py config.json')
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
    if 'sensitivity' in cfg:
      if 'GVrms' in cfg['sensitivity']:
        krms = cfg['sensitivity']['GVrms']['k']
      else:
        krms = None
      sens = pv3_fn.sensitivity_analysis (model, bPrint=True, bAutoRange=True, cfgKRMS = krms)
    else:
      sens = pv3_fn.sensitivity_analysis (model, bPrint=True, bAutoRange=True)
    print ('Maximum Norton Sensitivity = {:.6f}'.format (sens))
    clamp = pv3_fn.clamp_loss (model, bPrint=True)
    print ('Total Norton Clamping Loss = {:.6f}'.format (clamp))
  elif 'Id' in model.COL_U and 'Iq' in model.COL_U and 'Vd' in model.COL_Y and 'Vq' in model.COL_Y:
    sens = pv3_fn.thevenin_sensitivity_analysis (model, bPrint=True)
    print ('Maximum Thevenin Sensitivity = {:.6f}'.format (sens))
  elif 'Vdlo' in model.COL_U and 'Vqlo' in model.COL_U and 'Idlo' in model.COL_Y and 'Iqlo' in model.COL_Y: # unbalanced model
    sens = pv3_fn.sensitivity_analysis (model, bPrint=True, bAutoRange=True)
    print ('Maximum Norton Lo Sensitivity = {:.6f}'.format (sens))
  else:
    print ('No Thevenin or Norton columns found: skipping sensitivity analysis')
    #sens = sensitivity_analysis (model, bPrint=True, bLog=False, bAutoRange=True)
    #print ('Maximum Auto Sensitivity = {:.6f}'.format (sens))

  quit()

  #print ('model.clamps', model.clamps)
  if 'sensitivity' in cfg:
    print ('model.sensitivity', model.sensitivity)
    sens = pv3_fn.model_sensitivity (model, bPrint=True)
    print ('Maximum Sensitivity = {:.6f}'.format (sens))
    sens_loss = max (sens - model.sensitivity['limit'], 0.0)
    print ('Sensitivity Loss = {:.6f}'.format (sens_loss))
    rmse, mae, case_rmse, case_mae = model.trainingErrors(True)
    loss_rmse = model.n_cases * rmse * rmse
    print ('RMSE =', rmse, 'Loss RMSE =', loss_rmse)
    print ('Total Loss =', model.trainingLosses())

#  if case_idx < 0:
#    for idx in range(model.n_cases):
#      pv3_fn.analyze_case (model, idx, bPrint=False)
#  else:
#    pv3_fn.analyze_case (model, case_idx, bPrint=True)

#  print ('Rd range [{:.3f}-{:.3f}]'.format(minRd, maxRd))
#  print ('Rq range [{:.3f}-{:.3f}]'.format(minRq, maxRq))
#  print ('Event counts: G={:d} Ud={:d} Uq={:d} R={:d} Total={:d}'.format (nG, nUd, nUq, nRd, nG+nUd+nUq+nRd))
