# Copyright (C) 2018-2024 Battelle Memorial Institute
# file: test_clamp.py
""" Work on loss function for exceeding the clamp limits.

Paragraph.

Public Functions:
    :main: does the work
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.directory'] = os.getcwd()

if __name__ == '__main__':
  npts = 1001
  omega = 2*np.pi*60.0
  t=np.linspace(0.0, 0.2, npts)
  dt = 0.2 / (npts-1)
  y=np.sin(omega*t)
  lower = -0.8 * np.ones (npts)
  upper = 0.9 * np.ones (npts)
  zeros = np.zeros (npts)
  p1 = np.maximum (zeros, y - upper)
  p2 = np.maximum (zeros, lower - y)
  pt = p1 + p2
  loss = np.trapz(pt, dx=dt)
  fig, ax = plt.subplots(1, 1, sharex = 'col', figsize=(8,6), constrained_layout=True)
  ax.set_title ('Clamping Loss = {:.6f}'.format (loss))
  ax.plot (t, y, label='y')
  ax.plot (t, lower, label='lower')
  ax.plot (t, upper, label='upper')
  ax.plot (t, pt, label='penalty')
  ax.grid ()
  ax.legend (loc='lower right')
  ax.set_xlabel ('Time [s]')
  plt.show()
  plt.close()

