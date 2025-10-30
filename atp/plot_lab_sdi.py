import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

plt.rcParams['savefig.directory'] = os.getcwd()

filename = '/data/sdi_dc.hdf5'

df = None
store = pd.HDFStore(filename)
for key in store.keys():
  df1 = store.get(key)
  tstamps = df1.index
  t = tstamps - tstamps[0]
  tmax = (t[-1]-t[0]).total_seconds()
  n = len(t)
  dt = tmax/float(n-1)
  if df is None:
    print ('Creating key={:s}, Tmax={:.6f} seconds, dt={:.6f} microseconds for {:d} points'.format (key, tmax, dt*1.0e6, n))
    tbase = np.linspace (0.0, tmax, n)
    df = pd.DataFrame (data=df1[df1.columns[0]].to_numpy(), index=tbase, columns=[key])
  else:
    print ('  adding key{:s}'.format(key))
    df[key] = df1[df1.columns[0]].to_numpy()

print(df.describe())

ax = df.plot (title='SDI Test for DC Current', subplots=True, grid=True, xlabel='seconds')
plt.show()

# write these to blank-delimited trigger input files for LTSpice
spicepath = '/projects/ucf_invcontrol/lab/ltspice'
for trial in df.columns:
  fname = spicepath + trial + '.txt'
  fp = open (fname, 'w')
  print ('writing LTSpice source to', fname)
  x = df.index.to_numpy()
  y = df[trial].to_numpy()
  npts = len(x)
  for i in range(npts):
    print ('{:12.9f} {:12.9f}'.format(x[i], y[i]), file=fp)
  fp.close()
