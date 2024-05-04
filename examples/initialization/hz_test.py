# copyright 2022-2024 Battelle Memorial Institute
# demonstrating the efficient initialization for H(z)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

a=np.array([-0.26165449619293213, 0.1842983514070511])
b=np.array([0.16575227677822113, -0.04380539432168007, -0.21196550130844116])

if __name__ == '__main__':
  u0 = -0.2
  u1 = 1.0
  uh = np.zeros(3)
  yh = np.zeros(2)
  ustep = 1.0
  npts = 400
  y = np.zeros(npts)
  u = u0 * np.ones(npts)
  u[150:] = u1
  kss = np.sum(b) / (1.0 + np.sum(a))
  print ('Initial SS y={:.6f} for u={:.6f}'.format (kss*u0, u0))
  print ('Final   SS y={:.6f} for u={:.6f}'.format (kss*u1, u1))

  # output with no H(z) initialization
  for i in range(npts):
    uh[1:] = uh[:-1]
    uh[0] = u[i]
    ynew = np.sum(np.multiply(b, uh)) - np.sum(np.multiply(a, yh))
    yh[1:] = yh[:-1]
    yh[0] = ynew
    y[i] = ynew

  # output with no H(z) initialization
  uh[:] = u0
  yh[:] = kss * u0
  y_ic = np.zeros(npts)
  for i in range(npts):
    uh[1:] = uh[:-1]
    uh[0] = u[i]
    ynew = np.sum(np.multiply(b, uh)) - np.sum(np.multiply(a, yh))
    yh[1:] = yh[:-1]
    yh[0] = ynew
    y_ic[i] = ynew

  fig, ax = plt.subplots (2, 1, sharex = 'col', figsize=(12,9), constrained_layout=True)
  fig.suptitle ('H(z) Step Response from Rest')

  ax[0].plot (u, 'r', label='u')
  ax[1].plot (y, 'b', label='y_0')
  ax[1].plot (y_ic, 'r', label='y_ic')
  for j in range(2):
    ax[j].grid()
    ax[j].legend()

  plt.rcParams['savefig.directory'] = os.getcwd()
  plt.show()
  plt.close()

