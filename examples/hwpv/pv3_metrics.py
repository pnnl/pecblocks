# copyright 2021-2022 Battelle Memorial Institute
# summarizes RMSE and MAE for all training cases of an HW model for three-phase inverters
#  arg1: relative path to trained model configuration file
#  arg2: relative path to the training data file, HDF5

import os
import sys

import pv3_poly as pv3_model

data_path = 'c:/data/ucf2.hdf5'
model_path = './ucf2ac/ucf2ac_config.json'

if __name__ == '__main__':
  if len(sys.argv) > 1:
    model_path = sys.argv[1]
    if len(sys.argv) > 2:
      data_path = sys.argv[2]

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
  rmse, mae, case_rmse, case_mae = model.trainingErrors(True)
#  print ('Case RMSE and MAE:')
#  colstr = ' '.join('{:>6s}'.format(col) for col in model.COL_Y)
#  print ('Idx ', colstr, colstr)
  colstr = ','.join('{:s}'.format(col) for col in model.COL_Y)
  print ('Idx,{:s},{:s}'.format(colstr, colstr))
  h1str = ','.join('{:s}'.format('RMSE') for col in model.COL_Y)
  h2str = ','.join('{:s}'.format('MAE') for col in model.COL_Y)
  print (',{:s},{:s}'.format(h1str, h2str))
  valstr = ','.join('{:.4f}'.format(rmse[col]) for col in model.COL_Y)
  maestr = ','.join('{:.4f}'.format(mae[col]) for col in model.COL_Y)
  print ('Total,{:s},{:s}'.format(valstr, maestr))
  for i in range(len(case_rmse)):
#    valstr = ' '.join('{:6.4f}'.format(case_rmse[i][col]) for col in model.COL_Y)
#    maestr = ' '.join('{:6.4f}'.format(case_mae[i][col]) for col in model.COL_Y)
#    print ('{:3d}  {:s}  {:s}'.format(i, valstr, maestr))
    valstr = ','.join('{:.4f}'.format(case_rmse[i][col]) for col in model.COL_Y)
    maestr = ','.join('{:.4f}'.format(case_mae[i][col]) for col in model.COL_Y)
    print ('{:d},{:s},{:s}'.format(i, valstr, maestr))
  print ('Total Error Summary')
  for col in model.COL_Y:
    print ('{:4s} MAE={:8.4f} RMSE={:8.4f}'.format (col, mae[col], rmse[col]))

