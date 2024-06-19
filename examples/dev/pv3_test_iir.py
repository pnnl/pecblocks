# copyright 2021-2022 Battelle Memorial Institute
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

data_path = './data/balanced.hdf5'
model_path = './big/balanced_fhf.json'

def plot_case(model, idx):
  rmse, mae, y_hat, y_true, u = model.testOneCase(idx)
  y_iir = model.stepOneCase(idx)
  print ('column', model.COL_Y, 'RMS errors', rmse)
  valstr = ' '.join('{:.4f}'.format(rms) for rms in rmse)
#  print ('y_hat shape', y_hat.shape)
#  print ('y_true shape', y_true.shape)
#  print ('u shape', u.shape)

  fig, ax = plt.subplots (2, 4, sharex = 'col', figsize=(15,8), constrained_layout=True)
  fig.suptitle ('Case {:d} Simulation; Output RMSE = {:s}'.format(idx, valstr))
  for j in range(4):
    j1 = 2*j
    j2 = j1+1
    col1 = model.COL_U[j1]
    col2 = model.COL_U[j2]
    ax[0,j].set_title ('Inputs {:s}, {:s}'.format (col1, col2))
    ax[0,j].plot (model.t, u[:,j1], label=col1)
    ax[0,j].plot (model.t, u[:,j2], label=col2)
    ax[0,j].legend()
  j = 0
  for col in model.COL_Y:
    scale = model.normfacs[col]['scale']
    offset = model.normfacs[col]['offset']
    ax[1,j].set_title ('Output {:s}'.format (col))
    ax[1,j].plot (model.t, y_true[:,j]*scale + offset, label='y')
    ax[1,j].plot (model.t, y_hat[0,:,j]*scale + offset, label='y_dyno', linewidth=2)
    ax[1,j].plot (model.t, y_iir[:,j]*scale + offset, label='y_iir')
    ax[1,j].legend()
    j += 1
  plt.show()

if __name__ == '__main__':

  case_idx = 189
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
  if len(sys.argv) > 2:
    if int(sys.argv[2]) > 0:
      bNormalized = True
  if len(sys.argv) > 3:
    model_path = sys.argv[3]
  if len(sys.argv) > 4:
    data_path = sys.argv[4]

  model_folder, config_file = os.path.split(model_path)
  print ('model_folder =', model_folder)
  print ('data_path =', data_path)

  model = pv3_model.pv3()
  model.load_sim_config(model_path, model_only=False)
  model.loadTrainingData(data_path)
  model.applyNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()

  if case_idx < 0:
    for idx in range(model.n_cases):
      plot_case (model, idx)
  else:
    plot_case (model, case_idx)

