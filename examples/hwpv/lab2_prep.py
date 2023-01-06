import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
import numpy as np
from scipy import signal
import h5py
import glob

plt.rcParams['savefig.directory'] = os.getcwd()

SQRT2 = math.sqrt(2.0)
OMEGA = 2.0 * math.pi
# take the middle 0.3s, downsampled to 1 ms
TMAX = 0.3
DT = 0.001
PREFIX = 'scope'
FC = 60.0
VC = 120.0
# the scope file save accounts for 1x and 20x probes and the V/div
# these scale factors only have to account for the current probe 0.1V/A
SCALE_VDC = 1.0
SCALE_IDC = 10.0
SCALE_VAC = 1.0
SCALE_IAC = 10.0

tticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

input_path = 'c:/data/outback/outbac*.csv'
output_path = 'c:/data/lab2.hdf5'

# channel names
# Vdc = DCVoltage
# Idc = DCCurrent
# Vac = ACVoltage
# Iac = ACCurrent

b, a = signal.butter (2, 1.0 / 16384.0, btype='lowpass', analog=False)

def my_decimate(x, q, method='butter'):
  if method == 'fir':
    return signal.decimate (x, q, ftype='fir', n=None)
  elif method == 'butter':
    return signal.lfilter (b, a, x)[::q]
  elif method == 'slice':
    return x[::q]
  elif q == 65:  # downsampling 1 MHz signals to 256 samples per 60-Hz cycle
    return signal.decimate (signal.decimate(x, 5), 13)
  elif q <= 13:
    return signal.decimate (x, q)
  elif q == 800:
    return signal.decimate (signal.decimate (signal.decimate(x, 10), 10), 8)
  elif q == 1000:
    return signal.decimate (signal.decimate (signal.decimate(x, 10), 10), 10)
  return x[::q] # default will be slice

# create Vd, Vq, Id, Iq
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

if __name__ == '__main__':
  if len(sys.argv) > 1:
    input_path = sys.argv[1]
    if len(sys.argv) > 2:
      output_path = sys.argv[2]

  tbase = np.linspace (0.0, 0.6, 5000, endpoint=False)
  dt = tbase[1] - tbase[0]
  print ('tbase {:.8f} to {:.8f} at dt={:.8f}'.format (tbase[0], tbase[-1], dt))

  files = glob.glob (input_path)
  print ('Writing {:d} CSV files to {:s}'.format (len(files), output_path))
  idx = 1
  for fname in files:
    d = np.loadtxt (fname, delimiter=',', skiprows=1)
    t = d[:,0] - d[0,0]
    vdc = d[:,1] * SCALE_VDC
    idc = d[:,2] * SCALE_IDC
    vac = d[:,3] * SCALE_VAC
    iac = d[:,4] * SCALE_IAC
    n = len(t)
    dtrec = (t[-1] - t[0]) / float(n-1.0)
    if dtrec > dt:
#      print ('  interpolating on tbase')
      vdc = np.interp(tbase, t, vdc.copy())
      idc = np.interp(tbase, t, idc.copy())
      vac = np.interp(tbase, t, vac.copy())
      iac = np.interp(tbase, t, iac.copy())
#    print ('{:2d} dt={:.2f}us max=[{:.3f},{:.3f},{:.3f},{:.3f}] Tmax={:.6f}'.format (idx, dt*1.0e6,
#      np.max(np.abs(vdc)), np.max(np.abs(idc)), np.max(np.abs(vac)), np.max(np.abs(iac)), tbase[-1]))
    if idx == 18:
      fig, ax = plt.subplots (4, 1, sharex = 'col', figsize=(16,8), constrained_layout=True)
      fig.suptitle ('Case {:s}'.format (fname))
      ax[0].set_title('Vdc')
      ax[1].set_title('Idc')
      ax[2].set_title('Vac')
      ax[3].set_title('Iac')
      ax[0].plot(tbase, vdc)
      ax[1].plot(tbase, idc)
      ax[2].plot(tbase, vac)
      ax[3].plot(tbase, iac)
      for i in range(4):
        ax[i].grid()
        ax[i].set_xticks(tticks)
        ax[i].set_xlim (tticks[0], tticks[-1])
      plt.show()
    idx += 1
  quit()

  nbase = int(TMAX/DT)+1
  tbase = np.linspace (0.0, TMAX, nbase)
  Vc = VC * np.ones (nbase)
  Fc = FC * np.ones (nbase)
  #print (tbase.shape, Vc.shape, Fc.shape)

  store = pd.HDFStore(input_path)
  idx = 0
  f = h5py.File (output_path, 'w')
  #print (store.info())

  fig, ax = plt.subplots (2, 6, sharex = 'col', figsize=(16,8), constrained_layout=True)
  ax[0,0].set_title('Fc')
  ax[1,0].set_title('Vc')
  ax[0,1].set_title('Vdc')
  ax[1,1].set_title('Idc')
  ax[0,2].set_title('Vac')
  ax[1,2].set_title('Iac')
  ax[0,3].set_title('Vd')
  ax[1,3].set_title('Vq')
  ax[0,4].set_title('Id')
  ax[1,4].set_title('Iq')
  ax[0,5].set_title('Vrms')
  ax[1,5].set_title('Irms')

  for key in store.keys():
    df = store.get(key)
    tstamps = df.index
    t = tstamps - tstamps[0]
    tmax = (t[-1]-t[0]).total_seconds()
    n = len(t)
    dt = tmax/float(n-1)
    #print ('Key={:s}, Tmax={:.6f} seconds, dt={:.6f} microseconds for {:d} points'.format (key, tmax, dt*1.0e6, n))
    tpad = 0.5 * (tmax - TMAX)
    istart = int(tpad/dt)
    iend = istart + int(TMAX/dt)
    q = int(DT/dt)
    #print ('   Istart={:d}, Iend={:d}, qdec={:d}'.format (istart, iend, q))
    x = np.linspace (0.0, tmax, n)
    df['x'] = x
    df.set_index ('x', inplace=True)
    Vdc = df['DCVoltage'].to_numpy()
    Idc = df['DCCurrent'].to_numpy()
    Vac = df['ACVoltage'].to_numpy()
    Iac = df['ACCurrent'].to_numpy()

    Vdc = my_decimate (Vdc, q)[:nbase] * SCALE_VDC
    Idc = my_decimate (Idc, q)[:nbase] * SCALE_IDC
    Vac = my_decimate (Vac, q)[:nbase] * SCALE_VAC
    Iac = my_decimate (Iac, q)[:nbase] * SCALE_IAC
    Vd, Vq, Vrms = simulate_osg (tbase, Vac, Fc*OMEGA)
    Id, Iq, Irms = simulate_osg (tbase, Iac, Fc*OMEGA)

    ax[0,0].plot(tbase, Fc)
    ax[1,0].plot(tbase, Vc)
    ax[0,1].plot(tbase, Vdc)
    ax[1,1].plot(tbase, Idc)
    ax[0,2].plot(tbase, Vac)
    ax[1,2].plot(tbase, Iac)
    ax[0,3].plot(tbase, Vd)
    ax[1,3].plot(tbase, Vq)
    ax[0,4].plot(tbase, Id)
    ax[1,4].plot(tbase, Iq)
    ax[0,5].plot(tbase, Vrms)
    ax[1,5].plot(tbase, Irms)

    grp = f.create_group ('{:s}{:d}'.format (PREFIX, idx))
    idx += 1
    grp.create_dataset ('t', data=tbase, compression='gzip')
    grp.create_dataset ('Fc', data=Fc, compression='gzip')
    grp.create_dataset ('Vc', data=Vc, compression='gzip')
    grp.create_dataset ('Vdc', data=Vdc, compression='gzip')
    grp.create_dataset ('Idc', data=Idc, compression='gzip')
    grp.create_dataset ('Vac', data=Vac, compression='gzip')
    grp.create_dataset ('Iac', data=Iac, compression='gzip')
    grp.create_dataset ('Vd', data=Vd, compression='gzip')
    grp.create_dataset ('Vq', data=Vq, compression='gzip')
    grp.create_dataset ('Vrms', data=Vrms, compression='gzip')
    grp.create_dataset ('Id', data=Id, compression='gzip')
    grp.create_dataset ('Iq', data=Iq, compression='gzip')
    grp.create_dataset ('Irms', data=Irms, compression='gzip')
    #quit()

  f.close()
  plt.show()


