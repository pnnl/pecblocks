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
import torch

plt.rcParams['savefig.directory'] = os.getcwd()

if __name__ == '__main__':
  npts = 1001
  omega = 2*np.pi*60.0
  ll = -0.8
  ul = 0.9

  # numpy calculations
  t=np.linspace(0.0, 0.2, npts)
  dt = 0.2 / (npts-1)
  y=np.sin(omega*t)
  lower = ll * np.ones (npts)
  upper = ul * np.ones (npts)
  zeros = np.zeros (npts)
  p1 = np.maximum (zeros, y - upper)
  p2 = np.maximum (zeros, lower - y)
  pt = p1 + p2
  loss_tz = np.trapz(pt, dx=dt)
  loss_np = dt * np.sum(pt)

  # torch calculations
  y_torch = torch.from_numpy (y)
  lower_torch = ll * torch.ones(npts, requires_grad=True)
  upper_torch = ul * torch.ones(npts, requires_grad=True)
  zeros_torch = torch.zeros(npts, requires_grad=True)
  p1_torch = torch.maximum (zeros_torch, y_torch - upper_torch)
  p2_torch = torch.maximum (zeros_torch, lower_torch - y_torch)
  loss_torch = dt * torch.sum(p1_torch + p2_torch)

  fig, ax = plt.subplots(1, 1, sharex = 'col', figsize=(8,6), constrained_layout=True)
  ax.set_title ('Clamping Loss = {:.6f} (TZ), {:.6f} (NP sum), {:.6f} (TRCH)'.format (loss_tz, loss_np, loss_torch))
  ax.plot (t, y, label='y')
  ax.plot (t, lower, label='lower')
  ax.plot (t, upper, label='upper')
  ax.plot (t, pt, label='penalty')
  ax.grid ()
  ax.legend (loc='lower right')
  ax.set_xlabel ('Time [s]')
  plt.show()
  plt.close()

