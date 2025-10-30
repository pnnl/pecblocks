import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_hdf('./TACS_HWPV_Test.hdf5')
#df.info()
#print (df.describe(include='all'))

print ('Column                         Min           Max')
for lbl, data in df.iteritems():
  print ('{:20s} {:13.5f} {:13.5f}'.format (lbl, data.min(), data.max()))

#ax = df.plot (title='TACS HWPV Test', subplots=True)
#plt.show()

fig, ax = plt.subplots (3, 3, sharex = 'col', figsize=(18,8), constrained_layout=True)
fig.suptitle ('TACS HWPV Test')
t = df.index
tmin = t[0]
tmax = t[-1]
if len(sys.argv) > 2:
  tmin = float(sys.argv[1])
  tmax = float(sys.argv[2])
# scaled inputs
for col in ['V:G:', 'V:T:', 'V:FC:', 'V:MD:', 'V:MQ:', 'V:CTL:']:
  if col == 'V:G:':
    base = 1000.0
  elif col == 'V:T:':
    base = 50.0
  elif col == 'V:FC:':
    base = 60.0
  else:
    base = 1.0
  ax[0,0].plot (df.index, df[col]/base, label=col)

for col in ['V:PCCA', 'V:PCCB', 'V:PCCC']:
  ax[0,1].plot (t, df[col], label=col)
#for col in ['I:PCCA:LOAD A', 'I:PCCB:LOAD B', 'I:PCCC:LOAD C']:
#  ax[0,2].plot (t, df[col], label=col)
for col in ['T:VDC']:
  ax[1,0].plot (t, df[col], label=col)
for col in ['T:IDC']:
  ax[1,1].plot (t, df[col], label=col)
#for col in ['T:VRMS','T:VD','T:VQ','T:V0']:
#  ax[1,2].plot (t, df[col], label=col)
for col in ['T:ID','T:IDLO','T:IDHI']:
  ax[2,0].plot (t, df[col], label=col)
for col in ['T:IQ','T:IQLO','T:IQHI']:
  ax[2,1].plot (t, df[col], label=col)
#for col in ['T:I0LO','T:I0HI']:
#  ax[2,2].plot (t, df[col], label=col)

for col in ['V:PCCA', 'V:PCCB', 'V:PCCC']:
  ax[0,2].plot (t, df[col], label=col)
for col in ['I:PCCA:LOAD A', 'I:PCCB:LOAD B', 'I:PCCC:LOAD C']:
  ax[1,2].plot (t, df[col], label=col)
for col in ['T:VD','T:VQ','T:V0']:
  ax[2,2].plot (t, df[col], label=col)

for i in range(3):
  for j in range(3):
    ax[i,j].legend()
    ax[i,j].grid()
    ax[i,j].set_xlim (tmin, tmax)
  ax[i,2].set_xlim (8.9, 9.2)

plt.rcParams['savefig.directory'] = os.getcwd()
plt.show()
#plt.savefig('tacs_hwpv.png')

