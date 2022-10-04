import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import control
import harold

filters = {'pv1': {'L1':0.002, 'L2':0.0004, 'C':0.000020, 'dt':0.001, 'Tmax':0.2},
           'pv3': {'L1':0.00061, 'L2':0.0000367, 'C':0.0000191, 'dt':0.002, 'Tmax':0.2}}

aSTP=np.array([[  0.0,  0.0],
               [  0.1,  0.0],
               [0.001,  1.0],
               [999.0,  1.0]])

if __name__ == '__main__':
  lcl = filters['pv1']
  L1 = lcl['L1']
  L2 = lcl['L2']
  C = lcl['C']
  dt = lcl['dt']
  Tmax = lcl['Tmax']
  H11 = control.TransferFunction([-C, 0.0], [1.0], dt=0.0)
  H12 = control.TransferFunction([L1*C, 0.0, 1.0], [1.0], dt=0.0)
  print (lcl)
  print (H11)
  print (H12)
  num11 = (-2.0*C/dt) * np.array([1.0, -1.0])
  den11 = np.array([1.0, 1.0])
  Hz11 = control.TransferFunction (num11, den11, dt=dt)
  print (Hz11)
  K = 4.0 * L1 * C / dt / dt
  num12 = np.array([1+K, 2*(1-K), 1+K])
  den12 = np.array([1.0, 2.0, 1.0])
  Hz12 = control.TransferFunction (num12, den12, dt=dt)
  print (Hz12)

  npts = int(Tmax/dt) + 1
  t = np.linspace(0.0, Tmax, npts)
  y11 = np.zeros(npts)
  y12 = np.zeros(npts)
  u = np.interp(t, aSTP[:,0], aSTP[:,1])
  y11hist = np.zeros(len(den11))
  y12hist = np.zeros(len(den12))
  for i in range(2, npts):
    y11[i] = num11[0]*u[i] + num11[1]*u[i-1] - den11[1]*y11[i-1]
    y12[i] = num12[0]*u[i] + num12[1]*u[i-1] +num12[2]*u[i-2] - den12[1]*y11[i-1] - den12[2]*y11[i-2]

  fig, ax = plt.subplots (3, 1, sharex = 'col', figsize=(12,9), constrained_layout=True)
  fig.suptitle ('H2 Step Responses; L1 = {:.2f} mH, L2 = {:.2f} mH, C = {:.2f} uF, dt = {:.3f} ms'.format(L1*1000, L2*1000, C*1e6, dt*1000))

  ax[0].set_title ('Normalized Input')
  ax[0].plot (t, u, 'r', label='u')

  ax[1].set_title ('Normalized y11 (Vs-Irms)')
  ax[1].plot (t, y11, 'r', label='y11')

  ax[2].set_title ('Normalized y12 (Is-Irms)')
  ax[2].plot (t, y12, 'r', label='y12')

  for i in range(3):
    ax[i].set_xlim([t[0], t[-1]])
    ax[i].grid()
  plt.rcParams['savefig.directory'] = os.getcwd()
  plt.show()
  plt.close()

