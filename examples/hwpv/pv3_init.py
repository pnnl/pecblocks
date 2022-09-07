# copyright 2021-2022 Battelle Memorial Institute
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

model_path = './big/balanced_fhf.json'

def buildInputVector():
  tstop = 0.2
  dt = 0.002
  n = int(tstop/dt+1.0)
  t = np.linspace(0.0, tstop, num=n)
  G = np.full(n, 0.0)
  T = np.full(n, 25.0)
  Fc = np.full(n, 60.0)
  Md = np.full(n, 1.0)
  Mq = np.full(n, 0.0001)
  Vrms = np.full(n, 0.0)
  GVrms = np.full(n, 0.0)
  Ctl = np.full(n, 0.0)
  return t, G, T, Fc, Md, Mq, Vrms, GVrms, Ctl

def plot_case(model):
  t, G, T, Fc, Md, Mq, Vrms, GVrms, Ctl = buildInputVector()
  Vdc, Idc, Id, Iq = model.simulateVectors(G, T, Fc, Md, Mq, Vrms, GVrms, Ctl)

  fig, ax = plt.subplots (2, 4, sharex = 'col', figsize=(15,8), constrained_layout=True)
  fig.suptitle ('Initialization Test')
  inplots = [{'cols':['G','T'],'vals':[G,T]},
             {'cols':['Fc','Ctl'],'vals':[Fc,Ctl]},
             {'cols':['Md','Mq'],'vals':[Md,Mq]},
             {'cols':['Vrms','GVrms'],'vals':[Vrms,GVrms]}]
  j = 0
  for col in inplots:
    col1 = col['cols'][0]
    col2 = col['cols'][1]
    val1 = col['vals'][0]
    val2 = col['vals'][1]
    ax[0,j].set_title ('Inputs {:s}, {:s}'.format (col1, col2))
    ax[0,j].plot (t, val1, label=col1)
    ax[0,j].plot (t, val2, label=col2)
    ax[0,j].legend()
    j += 1

  ax[1,0].set_title('Output Vdc')
  ax[1,0].plot (t, Vdc)
  ax[1,1].set_title('Output Idc')
  ax[1,1].plot (t, Idc)
  ax[1,2].set_title('Output Id')
  ax[1,2].plot (t, Id)
  ax[1,3].set_title('Output Iq')
  ax[1,3].plot (t, Iq)

  plt.show()

if __name__ == '__main__':

  if len(sys.argv) > 1:
    model_path = sys.argv[1]

  model_folder, config_file = os.path.split(model_path)
  print ('model_folder =', model_folder)

  model = pv3_model.pv3()
  model.load_sim_config(model_path, model_only=False)
  model.batch_size = 1
  model.initializeModelStructure()
  model.loadModelCoefficients()

  plot_case (model)

