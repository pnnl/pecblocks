import os
import sys

import pv1_poly as pv1_model
#import pv1_fhf as pv1_model
#import pv1_model
#import pv1_feedback as pv1_model

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

if __name__ == '__main__':

  model = pv1_model.pv1(os.path.join(model_folder,'pv1_config.json'))
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization(os.path.join(model_folder,'normfacs.json'))
  model.initializeModelStructure()
  model.loadModelCoefficients(model_folder)
  rmse, mae, case_rmse = model.trainingErrors(True)
  print ('Case RMS Errors:')
  print ('Idx ', ' '.join('{:>6s}'.format(col) for col in model.COL_Y))
  for i in range(len(case_rmse)):
    valstr = ' '.join('{:6.4f}'.format(case_rmse[i][col]) for col in model.COL_Y)
    print ('{:3d}  {:s}'.format(i, valstr))
  print ('Total Error Summary')
  for col in model.COL_Y:
    print ('{:4s} MAE={:8.4f} RMSE={:8.4f}'.format (col, mae[col], rmse[col]))

