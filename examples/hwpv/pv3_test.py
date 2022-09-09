# copyright 2021-2022 Battelle Memorial Institute
# plots a comparison of simulated and true outputs from a trained HW model
#  arg1: case number to plot 1..ncases (default 189)
#  arg2: 1 to plot normalized quantities (default false)
#  arg3: relative path to trained model configuration file
#  arg4: relative path to the training data file, HDF5
#
# example: python pv3_test.py 189 1 flatbal/flatbal_config.json

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

nrows = 2
ncols = 9
bNormalized = False

data_path = './data/flatbalanced.hdf5'
model_path = './flatbal/flatbal_config.json'
report_path = './report'

def plot_case(model, idx, bPNG=False):
  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=500)
#  rmse, y_hat, y_true, u = model.stepOneCase(idx)
  if not bPNG:
    print ('column', model.COL_Y, 'RMS errors', rmse)
  valstr = ' '.join('{:.4f}'.format(rms) for rms in rmse)
  maestr = ' '.join('{:.4f}'.format(val) for val in mae)
#  print ('y_hat shape', y_hat.shape)
#  print ('y_true shape', y_true.shape)
#  print ('u shape', u.shape)

  i1 = 1 # 2*model.n_loss_skip

  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Case {:d} Simulation; Output RMSE = {:s}; Output MAE = {:s}'.format(idx, valstr, maestr))
  j = 0
  for key in model.COL_U:
    scale = model.normfacs[key]['scale']
    offset = model.normfacs[key]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    ax[0,j].set_title ('Input {:s}'.format (key))
    ax[0,j].plot (model.t[i1:], u[i1:,j]*scale + offset)
    j += 1
  j = 0
  for key in model.COL_Y:
    scale = model.normfacs[key]['scale']
    offset = model.normfacs[key]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    ax[1,j].set_title ('Output {:s}'.format (key))
    ax[1,j].plot (model.t[i1:], y_true[i1:,j]*scale + offset, label='y')
    ax[1,j].plot (model.t[i1:], y_hat[i1:,j]*scale + offset, label='y_hat')
    ax[1,j].legend()
    j += 1
  if bPNG:
    plt.savefig(os.path.join(report_path,'case{:d}.png'.format(idx)))
  else:
    plt.rcParams['savefig.directory'] = os.getcwd()
    plt.show()
  plt.close(fig)

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
      plot_case (model, idx, bPNG=True)
      if (idx+1) % 10 == 0:
        print ('plotted {:d} of {:d} cases'.format(idx+1, model.n_cases))
  else:
    plot_case (model, case_idx)

