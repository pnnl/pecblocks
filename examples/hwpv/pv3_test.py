# copyright 2021-2024 Battelle Memorial Institute
# plots a comparison of simulated and true outputs from a trained HW model
#  arg1: configuration file
#  arg2: case number to plot 1..ncases (default 189)
#  arg3: 1 to plot normalized quantities (default false)
#
# example: python pv3_test.py flatbal_config.json

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pecblocks.pv3_poly as pv3_model
import json

#nrows = 3 # 2
#ncols = 4 # 5 # 7 # 9
bNormalized = False

bWantMAE = False

#data_path = './data/flatbalanced.hdf5'
#model_path = './flatbal/flatbal_config.json'
#report_path = './report'
#
#model_path = './flatstable/flatstable_config.json'
#
#data_path = './data/balanced.hdf5'
#model_path = './big/balanced_config.json'
#
#data_path = '../../../atptools/unbalanced.hdf5'
#model_path = './tacs/tacs_config.json'
#model_path = './unbal/unbal_config.json'
#
#data_path = '../simscape/balanced.hdf5'
#model_path = '../simscape/balanced_config.json'
#
#data_path = './data/osg_vrms.hdf5'
#model_path = './osg_vrms/osg_vrms_config.json'
#
#data_path = './data/osg_vdvq.hdf5'
#model_path = './osg_vdvq/osg_vdvq_config.json'
#
#data_path = './data/balanced_vdvq.hdf5'
#model_path = './dc/dc_config.json'
#
#data_path = './data/flatbalanced.hdf5'
#model_path = './flatbal_continuation/flatbal_continuation_config.json'
#
#data_path = './data/osg_vdvq2.hdf5'
#model_path = './osg_vdvq/osg_vdvq_config.json'
#
#data_path = './data/balanced_vdvq2.hdf5'
#model_path = './flat_vdvq/flat_vdvq_config.json'
#
#data_path = 'c:/data/ucf2.hdf5'
#model_path = './ucf2/ucf2_config.json'
#model_path = './ucf2ac/ucf2ac_config.json'
#
#data_path = 'c:/data/sdi.hdf5'
#model_path = './sdi/sdi_config.json'
#
#data_path = 'c:/data/sdi4.hdf5'
#model_path = './sdi4/sdi4_config.json'
##model_path = './sdi4v/sdi4v_config.json'
#
#data_path = 'c:/data/sdi5.hdf5'
#model_path = './sdi5/sdi5_config.json'
#
#data_path = 'd:/data/unb4.hdf5'
#model_path = './unb4/unb4_config.json'
#
#data_path = 'd:/data/jan.hdf5'
#model_path = './jan/jan_config.json'
#
#data_path = 'd:/data/ucf2.hdf5'
#model_path = './ucf2ac/ucf2ac_config.json'
#
#data_path = 'd:/data/ucf3/ucf3z.hdf5'
#model_path = './ucf3z_config.json'
#
#data_path = 'd:/data/ucf3/ucf3.hdf5'
##model_path = './ucf3_config.json'
##model_path = './ucf4_config.json'
#model_path = './ucf6_config.json'
#
#data_path = 'd:/data/ucf3/ucf7.hdf5'
#model_path = './ucf7s_config.json'
#model_path = './ucf8_config.json'

#data_path = 'd:/data/osg4_vdvq.hdf5'
#model_path = './osg4/osg4_vdvq_config.json'

#data_path = 'd:/data/ucf3/ucf9.hdf5'
#model_path = './ucf9_config.json'
#model_path = './ucf10_config.json'

#data_path = 'd:/data/ucf3/ucf9c.hdf5'
#model_path = './ucf10c_config.json'

def plot_case(model, idx, bPNG=False):
#  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=500)
  rmse, mae, y_hat, y_true, u = model.stepOneCase(idx)
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

#  print ('y_hat shape', y_hat.shape)
#  print ('y_true shape', y_true.shape)
#  print ('u shape', u.shape)

  i1 = 1 # 2*model.n_loss_skip

  nrows = 3 # first two rows for inputs, last row for outputs
  ncols = len(model.COL_Y)
  while 2*ncols < len(model.COL_U):
    ncols += 1

  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Model {:s} Case {:d} Simulation; Output RMSE = {:s}, Output Y(0) = {:s}'.format(model.model_root, idx, valstr, icstr))
#  fig.suptitle ('Case {:d} Simulation; Output RMSE = {:s}; Output MAE = {:s}'.format(idx, valstr, maestr))
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
#    ax[row,col].ticklabel_format(useOffset=False)
    #print ('initial {:s}={:.6f}'.format (key, u[i1,j]*scale + offset))
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
#    ax[row,col].ticklabel_format(useOffset=False)
    print ('initial {:s}={:.6f}'.format (key, y_hat[i1,j]*scale + offset))
    j += 1
  if bPNG:
    plt.savefig(os.path.join(report_path,'case{:d}.png'.format(idx)))
  else:
    plt.rcParams['savefig.directory'] = os.getcwd()
    plt.show()
  plt.close(fig)

def add_case(model, idx, bTrueOutput=False):
  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=500)
  i1 = 1 # 2*model.n_loss_skip
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
    print ('Usage: python pv3_test.py config.json')
    quit()

  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)
  case_idx = 100 # 36 # 189
  if len(sys.argv) > 2:
    case_idx = int(sys.argv[2])
    if len(sys.argv) > 3:
      if int(sys.argv[3]) > 0:
        bNormalized = True

  model = pv3_model.pv3(training_config=config_file)
  model.loadTrainingData(data_path)
  model.loadAndApplyNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  print (len(model.COL_U), 'inputs:', model.COL_U)
  print (len(model.COL_Y), 'outputs:', model.COL_Y)

  if case_idx < 0:
    fig, ax = plt.subplots (2, 2, sharex = 'col', figsize=(12,8), constrained_layout=True)
    fig.suptitle ('Testing model {:s} estimated output with {:d} cases'.format(model_path, model.n_cases))
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
    fig.suptitle ('Testing model {:s} true output with {:d} cases'.format(model_path, model.n_cases))
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
    plot_case (model, case_idx)

