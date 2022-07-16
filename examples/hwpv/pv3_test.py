import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

root = 'unbalanced' # 'tacs' 'gfm8'
nrows = 2
ncols = 9

data_path = r'./data/{:s}.hdf5'.format(root)
#data_path = r'./data/unbalanced.hdf5'.format(root)
model_folder = r'./models'

def plot_case(model, idx):
  rmse, y_hat, y_true, u = model.testOneCase(idx)
#  rmse, y_hat, y_true, u = model.stepOneCase(idx)
  print ('column', model.COL_Y, 'RMS errors', rmse)
  valstr = ' '.join('{:.4f}'.format(rms) for rms in rmse)
#  print ('y_hat shape', y_hat.shape)
#  print ('y_true shape', y_true.shape)
#  print ('u shape', u.shape)

  i1 = 50 # 2*model.n_loss_skip

  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Case {:d} Simulation; Output RMSE = {:s}'.format(idx, valstr))
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
    ax[1,j].plot (model.t[i1:], y_hat[0,i1:,j]*scale + offset, label='y_hat')
    ax[1,j].legend()
    j += 1
  plt.show()

if __name__ == '__main__':

  case_idx = 189
  bNormalized = False
  if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
  if len(sys.argv) > 2:
    if int(sys.argv[2]) > 0:
      bNormalized = True

  model = pv3_model.pv3(os.path.join(model_folder,'{:s}_config.json'.format(root)))
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization(os.path.join(model_folder,'normfacs.json'))
  model.initializeModelStructure()
  model.loadModelCoefficients(model_folder)
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)

  if case_idx < 0:
    for idx in range(model.n_cases):
      plot_case (model, idx)
  else:
    plot_case (model, case_idx)

