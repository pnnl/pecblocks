import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
import os

plt.rcParams['savefig.directory'] = os.getcwd()
tags1 = ['Ia', 'Ib', 'Ic']
tags2 = ['Va', 'Vb', 'Vc']
tticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
tticks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#tticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

savePlot = False
SCALE_AMPS = 10.
SCOPE_IAC = 1
SCOPE_VAC = 2
SCOPE_DC = 3

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
#    print (key, df1.columns)
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

b, a = signal.butter (2, 1.0 / 64.0, btype='lowpass', analog=False)

for freq in [58, 60, 62]:
  for ud in [0.52, 0.56, 0.60, 0.64, 0.66, 0.68]:
#for freq in [60]:
#  for ud in [0.47]:
#for freq in [60]:
#  for ud in [0.66]:
    dfs = {}
    casename = 'GL_{:d}Hz_ud{:4.2f}'.format (int(freq), ud)
    for scope in [1, 2, 3]: # AC current, AC voltage, DC scopes.
      filename = '{:s}_PNNL_scope{:d}.hdf5'.format (casename, scope)
      print ('\nProcessing {:s}'.format(filename))
      dfs[scope] = get_npframes (filename)

    for key, df_iac in dfs[SCOPE_IAC].items(): # triggered events 0..7
      df_vac = dfs[SCOPE_VAC][key]
      df_dc = dfs[SCOPE_DC][key]
      # remove the DC offset from AC current
      ia_0 = np.mean(df_iac['Ia'])
      ib_0 = np.mean(df_iac['Ib'])
      ic_0 = np.mean(df_iac['Ic'])
      df_iac['Ia'] -= ia_0
      df_iac['Ib'] -= ib_0
      df_iac['Ic'] -= ic_0
      # calculate the AC and DC power
      ac_pwr = df_iac['Ia']*df_vac['Va'] + df_iac['Ib']*df_vac['Vb'] + df_iac['Ic']*df_vac['Vc']
      dc_pwr = df_dc['Idc']*df_dc['Vdc']
      # smooth the DC current and locate the trigger; in the first test set, Idc always steps up
      idc_flt = signal.filtfilt (b, a, df_dc['Idc'])[::1]
      dc_flt_pwr = idc_flt*df_dc['Vdc']

      fig, ax = plt.subplots (5, 1, sharex = 'col', figsize=(12,6), constrained_layout=True)
      fig.suptitle ('{:s} {:s}'.format(casename, key))
      ax[0].plot (df_vac['t'], ac_pwr, label='AC Power')
      ax[0].plot (df_dc['t'], dc_pwr, label='DC Power')
      ax[0].plot (df_dc['t'], dc_flt_pwr, label='DC Filtered Power')
      
      ax[1].plot (df_dc['t'], df_dc['Idc'], label='Idc raw')
      ax[1].plot (df_dc['t'], idc_flt, label='Idc filtered')
      # Vdc is very smooth
#      ax[1].plot (df_dc['t'], 0.01 * df_dc['Vdc'], label='Vdc/100')
      
      for idx in range(3):
        rescale = np.max(df_vac[tags2[idx]]) / np.max(df_iac[tags1[idx]])
        ax[idx+2].plot (df_iac['t'], rescale*df_iac[tags1[idx]], 
                        label='{:.2f}*{:s}'.format (rescale, tags1[idx]))
        ax[idx+2].plot (df_vac['t'], df_vac[tags2[idx]], label=tags2[idx])
      
      for row in range(5):
        ax[row].legend()
        ax[row].grid()
        ax[row].set_xticks (tticks)
        ax[row].set_xlim (tticks[0], tticks[-1])

#     fig, ax = plt.subplots (4, 2, sharex = 'col', figsize=(12,6), constrained_layout=True)
#     fig.suptitle ('{:s} {:s}'.format(casename, key))
#     ax[0,0].plot (df_iac['t'], df_iac['Trg'], label='Iac Trg')
#     ax[0,0].plot (df_vac['t'], df_vac['Trg'], label='Vac Trg')
#     ax[0,0].plot (df_dc['t'], df_dc['Trg'], label='DC Trg')
#     ax[0,0].plot (df_vac['t'], ac_pwr, label='AC Power')
#     ax[0,0].plot (df_dc['t'], dc_pwr, label='DC Power')
#
#     ax[0,1].plot (df_dc['t'], df_dc['Idc'], label='Idc raw')
#     ax[0,1].plot (df_dc['t'], idc_flt, label='Idc filtered')
#     ax[0,1].plot (df_dc['t'], 0.01 * df_dc['Vdc'], label='Vdc/100')
#
#     for idx in range(3):
#       ax[idx+1, 0].plot (df_iac['t'], df_iac[tags1[idx]], label=tags1[idx])
#       ax[idx+1, 1].plot (df_vac['t'], df_vac[tags2[idx]], label=tags2[idx])
#
#     for row in range(4):
#       for col in range(2):
#         ax[row,col].legend()
#         ax[row,col].grid()
#         ax[row,col].set_xticks (tticks)
#         ax[row,col].set_xlim (tticks[0], tticks[-1])

      if savePlot:
        png_name = 'sdi_{:s}.png'.format (key[8:])
        plt.savefig(png_name)
      else:
        plt.show()
      plt.close()
      quit()

