import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv1_poly as pv1_model
#import pv1_fhf as pv1_model
#import pv1_model
#import pv1_feedback as pv1_model

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

def plot_case(model, idx):
  rmse, y_hat, y_true, u = model.testOneCase(idx)
#  rmse, y_hat, y_true, u = model.stepOneCase(idx)
  print ('column', model.COL_Y, 'RMS errors', rmse)
  valstr = ' '.join('{:.4f}'.format(rms) for rms in rmse)
#  print ('y_hat shape', y_hat.shape)
#  print ('y_true shape', y_true.shape)
#  print ('u shape', u.shape)

  fig, ax = plt.subplots (2, 5, sharex = 'col', figsize=(15,8), constrained_layout=True)
  fig.suptitle ('Case {:d} Simulation; Output RMSE = {:s}'.format(idx, valstr))
  j = 0
  for col in model.COL_U:
    scale = model.normfacs[col]['scale']
    offset = model.normfacs[col]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    if j < 5:
      ax[0,j].set_title ('Input {:s}'.format (col))
      ax[0,j].plot (model.t, u[:,j]*scale + offset)
    else:
      ax[1,j-2].set_title ('Derived {:s}'.format (col))
      ax[1,j-2].plot (model.t, u[:,j]*scale + offset)
    j += 1
  j = 0
  for col in model.COL_Y:
    scale = model.normfacs[col]['scale']
    offset = model.normfacs[col]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    ax[1,j].set_title ('Output {:s}'.format (col))
    ax[1,j].plot (model.t, y_true[:,j]*scale + offset, label='y')
    ax[1,j].plot (model.t, y_hat[0,:,j]*scale + offset, label='y_hat')
#    ax[1,j].plot (model.t, y_hat[:,j]*scale + offset, label='y_hat')
    ax[1,j].legend()
    j += 1
  plt.show()

if __name__ == '__main__':

  case_idx = 36
  bNormalized = False
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
  if len(sys.argv) > 2:
    if int(sys.argv[2]) > 0:
      bNormalized = True

  model = pv1_model.pv1(os.path.join(model_folder,'pv1_config.json'))
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization(os.path.join(model_folder,'normfacs.json'))
  model.initializeModelStructure()
  model.loadModelCoefficients(model_folder)

  if case_idx < 0:
    for idx in range(model.n_cases):
      plot_case (model, idx)
  else:
    plot_case (model, case_idx)

