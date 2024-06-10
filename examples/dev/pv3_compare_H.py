# copyright 2021-2022 Battelle Memorial Institute
# plots a comparison of simulated and true outputs from three variants in HW model:
#  gtype=iir
#  gtype=fir
#  gtype=stable2nd
#
# example: python pv3_test.py 189 1 flatbal/flatbal_config.json

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

data_path = './data/flatbalanced.hdf5'
data_path = './data/balanced.hdf5'

models = [{'title':'IIR', 'path':'./flatbal/flatbal_config.json'},
          {'title':'FIR', 'path':'./flatfir/flatfir_config.json'},
          {'title':'2nd', 'path':'./flatstable/flatstable_config.json'}]
models = [{'title':'BAL', 'path':'./big/balanced_config.json'}]
cols = [{'index':678,'title':'G[1000=>600] T[15]', 'lbl':'G-'},
        {'index':3,'title':'G[100=>250] T[15]', 'lbl':'G+'},
        {'index':1280,'title':'G/T[800/35] Fc[60=>55]', 'lbl':'F-'},
        {'index':1289,'title':'G/T[800/35] Fc[60=>65]', 'lbl':'F+'},
        {'index':1290,'title':'G/T[800/35] Md[1=>0.8]', 'lbl':'Md-'},
        {'index':1309,'title':'G/T[800/35] Md[1=>1.2]', 'lbl':'Md+'},
        {'index':1310,'title':'G/T[800/35] Mq[0=>-0.5]', 'lbl':'Mq-'},
        {'index':1329,'title':'G/T[800/35] Mq[0=>0.5]', 'lbl':'Mq+'},
        {'index':1330,'title':'G/T[800/35] Rg[2.88=>3.60]', 'lbl':'Rg+'},
        {'index':1349,'title':'G/T[800/35] Rg[2.88=>2.40]', 'lbl':'Rg-'}]
rows = ['Vdc', 'Idc', 'Id', 'Iq']

if __name__ == '__main__':
  print ('common data_path =', data_path)

  nrows = len(rows)
  ncols = len(cols)
  plt.rc('font', family='serif')
  plt.rc('xtick', labelsize=8)
  plt.rc('ytick', labelsize=8)
  plt.rc('axes', labelsize=8)
  plt.rc('legend', fontsize=8)
  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Comparing H1 Implementations')

  bYtruePlotted = False
  for mdl in models:
    print ('loading {:s} from {:s}'.format (mdl['title'], mdl['path']))
    model = pv3_model.pv3(training_config=mdl['path'])
    model.loadTrainingData(data_path)
    model.loadAndApplyNormalization()
    model.initializeModelStructure()
    model.loadModelCoefficients()
    j = 0
    for col in cols:
      idx = col['index']
      lbl = col['lbl']
      rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=500)
      i = 0
      for row in rows:
        scale = model.normfacs[row]['scale']
        offset = model.normfacs[row]['offset']
        if not bYtruePlotted:
          ax[i,j].set_title ('{:s} {:s}'.format (lbl, row))
          ax[i,j].plot (model.t, y_true[:,i]*scale + offset, label='y')
        ax[i,j].plot (model.t, y_hat[:,i]*scale + offset, label=mdl['title'])
        i += 1
      j += 1
    bYtruePlotted = True

  for i in range(nrows):
    for j in range(ncols):
      ax[i,j].legend()
  plt.rcParams['savefig.directory'] = os.getcwd()
  if False:
    plt.savefig('compare_H1.pdf')
  plt.show()

