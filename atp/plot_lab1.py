import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

plt.rcParams['savefig.directory'] = os.getcwd()

SQRT2 = math.sqrt(2.0)
OMEGA = 2.0 * math.pi

def simulate_osg (t, v, wc):
  n = len(t)
  dt = t[1]-t[0]
  th = np.cumsum (wc) * dt
  sinth = np.sin(th)
  costh = np.cos(th)
  vd = np.zeros(n)
  vq = np.zeros(n)
  vp = 0.0
  x2 = 0.0
  qvp = 0.0
  for i in range(n):
    kveps = SQRT2 * (v[i] - vp)
    qveps = kveps - qvp
    dvp = qveps * wc[i]
    vp += dvp*dt
    x2 += vp*dt
    qvp = x2*wc[i]
    vd[i] = vp*costh[i] + qvp*sinth[i]
    vq[i] = -vp*sinth[i] + qvp*costh[i]
  vrms = np.sqrt(0.5*(vd*vd + vq*vq))
  return vd, vq, vrms

df = pd.read_hdf('/data/lab1.hdf5')
#df.info()
#print (df.describe(include='all'))

print ('Column                         Min           Max')
for lbl, data in df.iteritems():
  print ('{:20s} {:13.7f} {:13.7f}'.format (lbl, data.min(), data.max()))

#ax = df.plot (title='PV1 Test for Lab Data', subplots=True)
#plt.show()
#quit()

t = df.index
tmin = t[0]
tmax = t[-1]
if len(sys.argv) > 2:
  tmin = float(sys.argv[1])
  tmax = float(sys.argv[2])

vd, vq, vrms = simulate_osg (t, df['T:PCCV'].values, df['V:FCTRL'].values*OMEGA)
vd1, vq1, vrms1 = simulate_osg (t, df['V:LOAD:ACN'].values, df['V:FCTRL'].values*OMEGA)

fig, ax = plt.subplots (2, 2, sharex = 'col', figsize=(16,8), constrained_layout=True)
fig.suptitle ('PV1 Simulation of OSG for Lab Data Processing')

for col in ['V:LOAD:ACN', 'T:PCCV']:
  ax[0,0].plot (t, df[col], label=col)
for col in ['T:VOD', 'T:VOQ', 'T:VRMS']:
  ax[1,0].plot (t, df[col], label=col)
ax[0,1].plot (t, vd, label='Vd')
ax[0,1].plot (t, vq, label='Vq')
ax[0,1].plot (t, vrms, label='Vrms')
ax[1,1].plot (t, vd1, label='Vd1')
ax[1,1].plot (t, vq1, label='Vq1')
ax[1,1].plot (t, vrms1, label='Vrms1')

for i in range(2):
  for j in range(2):
    ax[i,j].legend()
    ax[i,j].grid()
    ax[i,j].set_xlim (tmin, tmax)

plt.show()

quit()

fig, ax = plt.subplots (2, 3, sharex = 'col', figsize=(16,8), constrained_layout=True)
fig.suptitle ('PV1 Test for Lab Data Processing')
# scaled inputs
for col in ['V:G', 'V:TEMP', 'V:FCTRL', 'V:UD', 'V:UQ']:
  if col == 'V:G':
    base = 1000.0
  elif col == 'V:TEMP':
    base = 50.0
  elif col == 'V:FCTRL':
    base = 60.0
  else:
    base = 1.0
  ax[0,0].plot (t, df[col]/base, label=col)

# intermediate outputs, 'T:INTWC'
for col in ['T:VP', 'T:IP']:
  ax[1,0].plot (t, df[col], label=col)

# voltages and currents
for col in ['V:DCP:DCN', 'I:RSEQ:DCP']:
  ax[0,1].plot (t, df[col], label=col)
for col in ['V:LOAD:ACN', 'T:PCCV']:
  ax[1,1].plot (t, df[col], label=col)
for col in ['I:ACP:LOAD', 'T:PCCI']:
  ax[0,2].plot (t, df[col], label=col)
for col in ['T:VOD', 'T:VOQ', 'T:VRMS']:
  ax[1,2].plot (t, df[col], label=col)

for i in range(2):
  for j in range(3):
    ax[i,j].legend()
    ax[i,j].grid()
    ax[i,j].set_xlim (tmin, tmax)
#  ax[i,2].set_xlim (8.9, 9.2)

plt.show()

