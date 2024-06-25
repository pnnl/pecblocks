# copyright 2021-2024 Battelle Memorial Institute
# plots a comparison of simulated and true outputs from a trained HW model
#  arg1: configuration file
#  arg2: case number to plot 0..ncases-1 (default 100)
#  arg3: plotting method, [0,1,2] (see usage)
#
# example: python pv3_test.py ucf3_config.json

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pecblocks.pv3_poly as pv3_model
import json

bNormalized = False

bWantMAE = False

def plot_case(model, idx, method=0, bPNG=False):
  if method == 0:
    rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=model.n_pad, bUseTorchDS=True)
  elif method == 1:
    rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=model.n_pad, bUseTorchDS=False)
  else: # method == 2
    rmse, mae, y_hat, y_true, u = model.stepOneCase(idx)
  print ('Variable Ranges for Case {:d}:'.format(idx))
  print ('Column       Min       Max      Mean     Range')
  col_idx = 0
  for c in model.COL_U + model.COL_Y:
    fac = model.normfacs[c]
    dmax = model.de_normalize (np.max (model.data_train[idx,:,col_idx]), fac)
    dmin = model.de_normalize (np.min (model.data_train[idx,:,col_idx]), fac)
    dmean = model.de_normalize (np.mean (model.data_train[idx,:,col_idx]), fac)
    drange = dmax - dmin
    if abs(drange) <= 0.0:
      drange = 1.0
    print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, dmin, dmax, dmean, drange))
    col_idx += 1
  if not bPNG:
    print ('column', model.COL_Y, 'RMS errors', rmse)
  valstr = ' '.join('{:.4f}'.format(rms) for rms in rmse)
  maestr = ' '.join('{:.4f}'.format(val) for val in mae)
  y_ic = []
  j = 0
  for key in model.COL_Y:
    y_ic.append (y_hat[1,j] * model.normfacs[key]['scale'] + model.normfacs[key]['offset'])
    j += 1
  icstr = ' '.join('{:.4f}'.format(val) for val in y_ic)

  i1 = 0 # 1 # 2*model.n_loss_skip

  nrows = 3 # first two rows for inputs, last row for outputs
  ncols = len(model.COL_Y)
  while 2*ncols < len(model.COL_U):
    ncols += 1

  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Model {:s} Case {:d} Simulation; Output RMSE = {:s}, Output Y(0) = {:s}'.format(model.model_root, idx, valstr, icstr))
  j = 0
  row = 0
  col = 0
  for key in model.COL_U:
    scale = model.normfacs[key]['scale']
    offset = model.normfacs[key]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    ax[row,col].set_title ('Input {:s}'.format (key))
    ax[row,col].plot (model.t[i1:], u[i1:,j]*scale + offset)
    ax[row,col].grid()
    j += 1
    col += 1
    if col >= ncols:
      col = 0
      row += 1
  j = 0
  for key in model.COL_Y:
    scale = model.normfacs[key]['scale']
    offset = model.normfacs[key]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    ax[2,j].set_title ('Output {:s}'.format (key))
    ax[2,j].plot (model.t[i1:], y_true[i1:,j]*scale + offset, label='y')
    ax[2,j].plot (model.t[i1:], y_hat[i1:,j]*scale + offset, label='y_hat')
    ax[2,j].legend()
    ax[2,j].grid()
    print ('initial {:s}={:.6f}'.format (key, y_hat[i1,j]*scale + offset))
    j += 1
  if bPNG:
    plt.savefig(os.path.join(report_path,'case{:d}.png'.format(idx)))
  else:
    plt.rcParams['savefig.directory'] = os.getcwd()
    plt.show()
  plt.close(fig)

def add_case(model, idx, bTrueOutput=False, method=0):
  if method == 0:
    rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=model.n_pad, bUseTorchDS=True, bLog=False)
  elif method == 1:
    rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=model.n_pad, bUseTorchDS=False, bLog=False)
  else: # method == 2
    rmse, mae, y_hat, y_true, u = model.stepOneCase(idx)
  i1 = 0 # 1 # 2*model.n_loss_skip
  for j in range(len(model.COL_Y)):
    key = model.COL_Y[j]
    scale = model.normfacs[key]['scale']
    offset = model.normfacs[key]['offset']
    if bNormalized:
      scale = 1.0
      offset = 0.0
    if bTrueOutput:
      ax[j // 2, j % 2].plot (model.t[i1:], y_true[i1:,j]*scale + offset, label='y')
    else:
      ax[j // 2, j % 2].plot (model.t[i1:], y_hat[i1:,j]*scale + offset, label='y_hat')

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
    print ('Usage: python pv3_test.py config.json [idx=100] [method=0]')
    print ('  idx is the 0-based case number to plot, or -1 to plot all')
    print ('  method is the RMSE evaluation method to apply:')
    print ('    0 - Use the Torch Dataloader. RMSE will match training results, but may not start smoothly.')
    print ('    1 - Initialize by pre-padding inputs. RMSE may not match, but initialization improves.')
    print ('    2 - Initialize by coefficients of H1(z), as in other simulators. RMSE may not match, but initialization improves.')
    print ('    method 2 evaluates RMSE over just the plotted range. methods 0 and 1 include RMSE over a pre-initialization period.')
    quit()

  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)
  case_idx = 100
  method = 0
  if len(sys.argv) > 2:
    case_idx = int(sys.argv[2])
    if len(sys.argv) > 3:
      method = int(sys.argv[3])

  model = pv3_model.pv3(training_config=config_file)
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)

  if case_idx < 0:
    fig, ax = plt.subplots (2, 2, sharex = 'col', figsize=(12,8), constrained_layout=True)
    fig.suptitle ('Testing model {:s} estimated output with {:d} cases'.format(model_root, model.n_cases))
    for j in range(len(model.COL_Y)):
      ax[j // 2, j % 2].set_title ('Estimated {:s}'.format (model.COL_Y[j]))
      ax[j // 2, j % 2].grid()
    for idx in range(model.n_cases):
      add_case (model, idx)
      if (idx+1) % 10 == 0:
        print ('plotted {:d} of {:d} cases'.format(idx+1, model.n_cases))
    plt.rcParams['savefig.directory'] = os.getcwd()
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots (2, 2, sharex = 'col', figsize=(12,8), constrained_layout=True)
    fig.suptitle ('Testing model {:s} true output with {:d} cases'.format(model_root, model.n_cases))
    for j in range(len(model.COL_Y)):
      ax[j // 2, j % 2].set_title ('True {:s}'.format (model.COL_Y[j]))
      ax[j // 2, j % 2].grid()
    for idx in range(model.n_cases):
      add_case (model, idx, bTrueOutput=True)
      if (idx+1) % 10 == 0:
        print ('plotted {:d} of {:d} cases'.format(idx+1, model.n_cases))
    plt.rcParams['savefig.directory'] = os.getcwd()
    plt.show()
    plt.close(fig)
  else:
    plot_case (model, case_idx, method=method)

