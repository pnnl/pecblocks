import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rcParams['savefig.directory'] = 'C:/projects/ucf_invcontrol/reports/images' # os.getcwd()

def make_loss_plot (title, defs):
  plt.rc('font', family='serif')
  plt.rc('xtick', labelsize=12)
  plt.rc('ytick', labelsize=12)
  plt.rc('axes', labelsize=12)
  plt.rc('legend', fontsize=12)

  fig, ax = plt.subplots(1, 3, sharex = 'col', figsize=(15,6), constrained_layout=True)
  fig.suptitle (title, fontsize=16)
  for i in range(len(defs)):
    ax[i].set_title (defs[i]['title'])
    ax[i].set_ylabel ('Log10')
    ax[i].set_xlabel ('Epoch')
    ax[i].plot(np.log10(defs[i]['data'][0]), label='Training Loss')
    ax[i].plot(np.log10(defs[i]['data'][1]), label='Validation Loss')
    ax[i].legend()
    ax[i].grid()
  plt.show()
  plt.close()

if __name__ == '__main__':
  big3 = np.load ('c:/data/big3/loss.npy')
  unb3 = np.load ('c:/src/pecblocks/examples/hwpv/unb3/loss.npy')
  ucf = np.load ('c:/src/pecblocks/examples/hwpv/ucf2ac/loss.npy')
  osg = np.load ('c:/src/pecblocks/examples/hwpv/osg4_vdvq/loss.npy')
  lab1 = np.load ('c:/src/pecblocks/examples/hwpv/lab1/loss.npy')
  lab2 = None

  plots3 = [{'title':'GFM Balanced', 'data': big3},
            {'title':'GFM Unbalanced', 'data': unb3},
            {'title':'GridLink', 'data': ucf}]
  make_loss_plot ('Three-phase Inverter HWPV Training', plots3)

  plots1 = [{'title':'OSG Switching', 'data': osg},
            {'title':'Lab1', 'data': lab1}]
  make_loss_plot ('Single-phase Inverter HWPV Training', plots1)

