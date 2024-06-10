import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
import numpy as np
from scipy import signal
import h5py
import glob

SHOW_PLOTS = False
USE_RGRID = False

plt.rcParams['savefig.directory'] = os.getcwd()
idc_b, idc_a = signal.butter (2, 1.0 / 64.0, btype='lowpass', analog=False)

PREFIX = 'scope'
SQRT2 = math.sqrt(2.0)
SQRT6 = math.sqrt(6.0)
OMEGA = 2.0 * math.pi
SDI_VDC = 600.0
SDI_VLLNOM = 367.4
SDI_RPAR = 330.0
SCALE_AMPS = 10.0

SCOPE_IAC = 1
SCOPE_VAC = 2
SCOPE_DC = 3

tticks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#tticks = [0.0, 0.05, 0.10]

input_path = 'd:/data/ken2/'
output_path = 'd:/data/sdi5a.hdf5'

input_path = 'd:/data/ken2/power_factor/'
output_path = 'd:/data/sdi5b.hdf5'

input_path = 'd:/data/unb4/'
output_path = 'd:/data/unb4.hdf5'

dec_b, dec_a = signal.butter (2, 1.0 / 256.0, btype='lowpass', analog=False)

def my_decimate(x, q, method='butter'):
  if method == 'fir':
    return signal.decimate (x, q, ftype='fir', n=None)
  elif method == 'butter':
    return signal.lfilter (dec_b, dec_a, x)[::q]
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

def getVll (ud, uq):
  return SDI_VLLNOM * math.sqrt(ud*ud + uq*uq)

def getRgrid (vll, pstep):
  if pstep <= 0.0:
    return SDI_RPAR
  Rstep = vll * vll / pstep
  return 1.0 / (1.0/Rstep + 1.0/SDI_RPAR)

def get_channel_parameters (col):
  if col == 'V_DC':
    return 'Vdc', 1.0
  elif col == 'A_V':
    return 'Va', 1.0
  elif col == 'B_V':
    return 'Vc', 1.0
  elif col == 'C_V':
    return 'Vb', 1.0
  elif col == 'I_DC':
    return 'Idc', SCALE_AMPS
  elif col == 'A_C':
    return 'Ia', -SCALE_AMPS
  elif col == 'B_C':
    return 'Ic', -SCALE_AMPS
  elif col == 'C_C':
    return 'Ib', -SCALE_AMPS
  elif col == 'Ard':
    return 'Trg', 1.0

def get_npframes (filename):
  dfs = {}
  store = pd.HDFStore(filename)
  for key in store.keys():
    df1 = store.get(key)
    tstamps = df1.index
    t = tstamps - tstamps[0]
    tmax = (t[-1]-t[0]).total_seconds()
    n = len(t)
    dt = tmax/float(n-1)
    tbase = np.linspace (0.0, tmax, n)
    dfs[key] = {}
    dfs[key]['t'] = tbase
    for col in df1.columns:
      tag, scale = get_channel_parameters (col)
      dfs[key][tag] = scale * df1[col].to_numpy()
  store.close()
  return dfs

# create Vd, Vq, Vrms, Id, Iq, Irms
def simulate_pll (t, wc, va, vb, vc, ia, ib, ic):
  n = len(t)
  dt = t[1]-t[0]
  Vd = np.zeros(n)
  Vq = np.zeros(n)
  Id = np.zeros(n)
  Iq = np.zeros(n)
  th = 0.0
  vq_hist = 0.0  # for PI feedback driving Vq to zero
  for i in range(n):
    # Clarke and Park transforms with existing angle
    valpha = (2.0*va[i]-vb[i]-vc[i])/SQRT6/SDI_VLLNOM
    vbeta = (vb[i]-vc[i])/SQRT2/SDI_VLLNOM
    ialpha = (2.0*ia[i]-ib[i]-ic[i])/SQRT6
    ibeta = (ib[i]-ic[i])/SQRT2
    sinth = math.sin(th)
    costh = math.cos(th)
    Vd[i] = valpha*costh + vbeta*sinth
    Vq[i] = -valpha*sinth + vbeta*costh
    Id[i] = ialpha*costh + ibeta*sinth
    Iq[i] = -ialpha*sinth + ibeta*costh
    # synchronization of angle for the next time step
    dw = 188.5*Vq[i] + 134000.0 * dt * (Vq[i] - vq_hist)
    vq_hist = Vq[i]
    we = wc[0] - dw
    th += we * dt
  Vd *= SDI_VLLNOM
  Vq *= SDI_VLLNOM
  Vrms = np.sqrt(Vd*Vd + Vq*Vq)
  Irms = np.sqrt(Id*Id + Iq*Iq)
  return Vd, Vq, Vrms, Id, Iq, Irms

def control_signals (ud, uq, hz, triggerKey, t1, t2, tbase):
  vll = getVll(ud, uq)
  idx = int(triggerKey[-1])
  p1 = 200.0 * (idx)
  p2 = p1 + 200.0
  r1 = getRgrid (vll, p1)
  r2 = getRgrid (vll, p2)
  slope = (r2-r1) / (t2-t1)

  n = len(tbase)
  fc = hz * np.ones(n)
  udc = ud * np.ones(n)
  uqc = uq * np.ones(n)
  vdc = SDI_VDC * np.ones(n)
  rc = 0.5 * (r1+r2) * np.ones(n)
  for i in range(n):
    if tbase[i] > t2:
      rc[i] = r2
    elif tbase[i] < t1:
      rc[i] = r1
    else:
      rc[i] = r1 + slope * (tbase[i] - t1)
  return fc, udc, uqc, rc, vdc

if __name__ == '__main__':
  print ('Writing HWPV training records {:s}'.format (output_path))
  f = h5py.File (output_path, 'w')
  tdec = 1.0 + np.linspace (0.0, 3.0, 3001)
  dt = tdec[1] - tdec[0]
  print ('tdec {:.8f} to {:.8f} at dt={:.8f}'.format (tdec[0], tdec[-1], dt))
  idx = 0

# Balanced test, Uq=0, 144 records
#  fixed_uq = 0.0
#  for freq in [58, 60, 62]:
#    for ud in [0.52, 0.56, 0.60, 0.64, 0.66, 0.68]:
# Balanced test, Uq=0.47, 8 records
  fixed_uq = 0.0 # 0.47
  for freq in [60]: # [58, 60, 62]: # [60]:
    for ud in [0.66]: # [0.52, 0.56, 0.60, 0.64, 0.66, 0.68]: # [0.66]:
      casename = 'GL_{:d}Hz_ud{:4.2f}'.format (int(freq), ud)

      dfs = {}
      for scope in [1, 2, 3]: # AC current, AC voltage, DC scopes.
        filename = '{:s}_unbalanced_scope{:d}.hdf5'.format (casename, scope)
        print ('\nProcessing {:s}'.format(input_path+filename))
        dfs[scope] = get_npframes (input_path+filename)

      for key, df_iac in dfs[SCOPE_IAC].items(): # triggered events 0..7
        df_vac = dfs[SCOPE_VAC][key]
        df_dc = dfs[SCOPE_DC][key]
        t = df_dc['t']

        # smooth the DC current, estimate a functional trigger location for Rc
        idc_flt = signal.filtfilt (idc_b, idc_a, df_dc['Idc'])[::1]
        idc_pre = np.mean(idc_flt[:30000])
        idc_post = np.mean(idc_flt[-30000:])
        ntrig1 = int(np.argwhere (idc_flt < idc_pre)[-1])
        ntrig2 = np.argmax (idc_flt > idc_post)
        ttrig1 = t[ntrig1]
        ttrig2 = t[ntrig2]

        # remove DC offsets from AC quantities, generate the dq components
        ia_0 = np.mean(df_iac['Ia'])
        ib_0 = np.mean(df_iac['Ib'])
        ic_0 = np.mean(df_iac['Ic'])
        fc, udc, uqc, rc, vdc = control_signals (ud, fixed_uq, freq, key, 
                                                 t1=min(ttrig1, ttrig2), t2=max(ttrig1, ttrig2), tbase=t)
        df_iac['Ia'] -= ia_0
        df_iac['Ib'] -= ib_0
        df_iac['Ic'] -= ic_0
        vdc = df_dc['Vdc']
        if USE_RGRID:
          va = df_iac['Ia'] * rc
          vb = df_iac['Ib'] * rc
          vc = df_iac['Ic'] * rc
        else:
          va = df_vac['Va']
          vb = df_vac['Vb']
          vc = df_vac['Vc']
        wc = OMEGA * fc
        Vd, Vq, Vrms, Id, Iq, Irms = simulate_pll (t, wc, va, vb, vc, df_iac['Ia'], df_iac['Ib'], df_iac['Ic'])
        # generate AC and DC power signals
        ac_pwr = df_iac['Ia']*df_vac['Va'] + df_iac['Ib']*df_vac['Vb'] + df_iac['Ic']*df_vac['Vc']
        dc_pwr = df_dc['Idc']*df_dc['Vdc']
        dc_flt_pwr = idc_flt*df_dc['Vdc']

        # create downsampled channels for HWPV training
        Vdc_dec = my_decimate (vdc, 20)[1000:4001] # np.interp(tdec, t, vdc.copy())
        Idc_dec = my_decimate (df_dc['Idc'], 20)[1000:4001] # np.interp(tdec, t, df['Idc'].copy())
        Vrms_dec = my_decimate (Vrms, 20)[1000:4001] # np.interp(tdec, t, Vrms.copy())
        Irms_dec = my_decimate (Irms, 20)[1000:4001] # np.interp(tdec, t, Irms.copy())
        Fc_dec = np.interp(tdec, t, fc.copy())
        Rc_dec = np.interp(tdec, t, rc.copy())
        Ud_dec = np.interp(tdec, t, udc.copy())
        Uq_dec = np.interp(tdec, t, uqc.copy())
        Vd_dec = my_decimate (Vd, 20)[1000:4001] # np.interp(tdec, t, Vd.copy())
        Vq_dec = my_decimate (Vq, 20)[1000:4001] # np.interp(tdec, t, Vq.copy())
        Id_dec = my_decimate (Id, 20)[1000:4001] # np.interp(tdec, t, Id.copy())
        Iq_dec = my_decimate (Iq, 20)[1000:4001] # np.interp(tdec, t, Iq.copy())
        grp = f.create_group ('{:s}{:d}'.format (PREFIX, idx))
        grp.create_dataset ('t', data=tdec-tdec[0], compression='gzip')
        grp.create_dataset ('Fc', data=Fc_dec, compression='gzip')
        grp.create_dataset ('Ud', data=Ud_dec, compression='gzip')
        grp.create_dataset ('Uq', data=Uq_dec, compression='gzip')
        grp.create_dataset ('Rc', data=Rc_dec, compression='gzip')
        grp.create_dataset ('Vdc', data=Vdc_dec, compression='gzip')
        grp.create_dataset ('Idc', data=Idc_dec, compression='gzip')
        grp.create_dataset ('Vd', data=Vd_dec, compression='gzip')
        grp.create_dataset ('Vq', data=Vq_dec, compression='gzip')
        grp.create_dataset ('Vrms', data=Vrms_dec, compression='gzip')
        grp.create_dataset ('Id', data=Id_dec, compression='gzip')
        grp.create_dataset ('Iq', data=Iq_dec, compression='gzip')
        grp.create_dataset ('Irms', data=Irms_dec, compression='gzip')
        idx += 1

        print ('  Trigger {:d}, Idc pre/post = [{:.4f},{:.4f}], trigger times = [{:.4f},{:.4f}]'.format(int(key[-1]), idc_pre, idc_post, ttrig1, ttrig2))
        print ('    DC offsets in AC current = [{:.4f},{:.4f},{:.4f}], Rc=[{:.3f},{:.3f}], Vrms={:.3f}, Irms={:.4f}'.format (ia_0, ib_0, ic_0, rc[0], rc[-1], np.mean(Vrms), np.mean(Irms)))

        if SHOW_PLOTS:
          fig, ax = plt.subplots (5, 3, sharex = 'col', figsize=(16,10), constrained_layout=True)
          fig.suptitle ('{:s} {:s} Use Rgrid={:s}'.format(casename, key, str(USE_RGRID)))
          ax[0,0].plot (t, df_idc['Idc'], label='inst')
          ax[0,0].plot (t, idc_flt, label='filtered', color='magenta')
          ax[0,0].plot ([ttrig1, ttrig2], [idc_pre, idc_post], marker='o', color='orange', linestyle='dotted', label='trigger')
          ax[0,0].plot (tdec, Idc_dec, label='decimated', color='red')
          ax[0,0].set_ylabel ('Idc')

          ax[0,1].plot (t, udc, label='Ud inst')
          ax[0,1].plot (tdec, Ud_dec, color='red', label='Ud dec')
          ax[0,1].plot (tdec, Uq_dec, color='blue', label='Uq dec')
          ax[0,1].set_ylabel ('Ud/Uq')
          ax[1,0].plot (t, rc, label='inst')
          ax[1,0].plot (tdec, Rc_dec, color='red', label='dec')
          ax[1,0].set_ylabel ('Rc')
          ax[1,1].plot (t, fc, label='inst')
          ax[1,1].plot (tdec, Fc_dec, color='red', label='dec')
          ax[1,1].set_ylabel ('Fc')
          ax[2,0].plot (t, df_iac['Ib'], color='red', label='Ib')
          ax[2,0].plot (t, df_iac['Ic'], color='blue', label='Ic')
          ax[2,0].plot (t, df_iac['Ia'], color='black', label='Ia') # black on top
          ax[2,0].set_ylabel ('Ia')
          ax[2,1].plot (t, vb, color='red', label='Vb')
          ax[2,1].plot (t, vc, color='blue', label='Vc')
          ax[2,1].plot (t, va, color='black', label='Va') # black on top
          ax[2,1].set_ylabel ('Va')
          ax[3,0].plot (t, df_iac['Ib'], color='red', label='Ib')
          ax[3,0].set_ylabel ('Ib')
          ax[3,1].plot (t, vb, color='red', label='Vb')
          ax[3,1].set_ylabel ('Vb')
          ax[4,0].plot (t, df_iac['Ic'], color='blue', label='Ic')
          ax[4,0].set_ylabel ('Ic')
          ax[4,1].plot (t, vc, color='blue', label='Vc')
          ax[4,1].set_ylabel ('Vc')
          ax[0,2].plot (t, vdc, label='inst')
          ax[0,2].plot (tdec, Vdc_dec, color='red', label='dec')
          ax[0,2].set_ylabel ('Vdc')
          ax[1,2].plot (t, Vd, label='Vd')
          ax[1,2].plot (t, Vrms, label='Vrms')
          ax[1,2].plot (tdec, Vd_dec, color='red', label='Vd dec')
          ax[1,2].plot (tdec, Vrms_dec, color='magenta', label='Vrms dec')
          ax[1,2].set_ylabel ('Vd/Vrms')
          ax[2,2].plot (t, Vq, label='Vq')
          ax[2,2].plot (tdec, Vq_dec, color='red', label='Vq dec')
          ax[2,2].set_ylabel ('Vq')
          ax[3,2].plot (t, Id, label='Id')
          ax[3,2].plot (t, Irms, label='Irms')
          ax[3,2].plot (tdec, Id_dec, color='red', label='Id dec')
          ax[3,2].plot (tdec, Irms_dec, color='magenta', label='Irms dec')
          ax[3,2].set_ylabel ('Id/Irms')
          ax[4,2].plot (t, Iq, label='Iq')
          ax[4,2].plot (tdec, Iq_dec, color='red', label='Iq dec')
          ax[4,2].set_ylabel ('Iq')

          for col in range(3):
            ax[4,col].set_xlabel('t [s]')
            for row in range(5):
              ax[row,col].grid()
              ax[row,col].legend()
              ax[row,col].set_xticks (tticks)
              ax[row,col].set_xlim (tticks[0], tticks[-1])

          plt.show()
          plt.close()
  f.close()

