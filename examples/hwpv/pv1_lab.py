import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['savefig.directory'] = os.getcwd()

store = pd.HDFStore('c:/data/foobar.hdf5')
#print (store.info())
for key in store.keys():
  df = store.get(key)
  tstamps = df.index
  t = tstamps - tstamps[0]
  tmax = (t[-1]-t[0]).total_seconds()
  n = len(t)
  dt = tmax/float(n-1)
  print ('Key={:s}, Tmax={:.6f} seconds, dt={:.6f} microseconds for {:d} points'.format (key, tmax, dt*1.0e6, n))
  x = np.linspace (0.0, tmax, n)
  df['x'] = x
  df.set_index ('x', inplace=True)

#  print (df.info())

  df.plot(subplots=True, title='key {:s} at {:s}'.format(key, str(tstamps[0])))
  plt.show()
  plt.close()

