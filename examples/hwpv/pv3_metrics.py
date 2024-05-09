# copyright 2021-2024 Battelle Memorial Institute
# summarizes RMSE and MAE for all training cases of an HW model for three-phase inverters
#  arg1: relative path to the model configuration file

import os
import sys
import json

import pecblocks.pv3_poly as pv3_model

bWantMAE = False

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
    print ('Usage: python pv3_metrics.py config.json')
    quit()

  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)

  model = pv3_model.pv3(training_config=config_file)
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  rmse, mae, case_rmse, case_mae = model.trainingErrors(True)
  out_size = len(model.COL_Y)
  colstr = ','.join('{:s}'.format(col) for col in model.COL_Y)
  h1str = ','.join('{:s}'.format('RMSE') for col in model.COL_Y)
  valstr = ','.join('{:.4f}'.format(rmse[j]) for j in range(out_size))
  if bWantMAE:
    h2str = ','.join('{:s}'.format('MAE') for col in model.COL_Y)
    maestr = ','.join('{:.4f}'.format(mae[j]) for j in range(out_size))
    print ('Idx,{:s},{:s}'.format(colstr, colstr))
    print ('#,{:s},{:s}'.format(h1str, h2str))
    print ('Total,{:s},{:s}'.format(valstr, maestr))
    for i in range(len(case_rmse)):
      valstr = ','.join('{:.4f}'.format(case_rmse[i][j]) for j in range(out_size))
      maestr = ','.join('{:.4f}'.format(case_mae[i][j]) for j in range(out_size))
      print ('{:d},{:s},{:s}'.format(i, valstr, maestr))
    print ('Total Error Summary')
    for j in range(out_size):
      print ('{:4s} MAE={:8.4f} RMSE={:8.4f}'.format (model.COL_Y[j], mae[j], rmse[j]))
  else:
    print ('Idx,{:s}'.format(colstr))
    print ('#,{:s}'.format(h1str))
    print ('Total,{:s}'.format(valstr))
    for i in range(len(case_rmse)):
      valstr = ','.join('{:.4f}'.format(case_rmse[i][j]) for j in range(out_size))
      print ('{:d},{:s}'.format(i, valstr))
    print ('Highest RMSE Cases')
    for j in range(out_size):
      mval = 0.0
      idx = 0
      nmax = 0
      for i in range(len(case_rmse)):
        val = case_rmse[i][j]
        if val > 0.05:
          nmax += 1
        if val > mval:
          idx = i
          mval = val
      print ('{:4s} Max RMSE={:8.4f} at Case {:d}; {:d} > 0.05'.format (model.COL_Y[j], mval, idx, nmax))
    print ('Total Error Summary')
    for j in range(out_size):
      print ('{:4s} RMSE={:8.4f}'.format (model.COL_Y[j], rmse[j]))
