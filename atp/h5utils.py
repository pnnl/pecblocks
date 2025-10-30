# Copyright (C) 2018-2022 Battelle Memorial Institute
# file: h5utils.py
""" Loads ATP simulation results from COMTRADE files, saves to HDF.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
from comtrade import Comtrade
import numpy as np
import h5py
import math
from scipy import signal

#print (plt.gcf().canvas.get_supported_filetypes())
#quit()

# for decimation
b, a = signal.butter (2, 1.0 / 4096.0, btype='lowpass', analog=False)

def my_decimate(x, q, method):
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

def decimate_channels(channels, k=1000, method='iir', filtered=[]): # can choose iir, fir, slice
  for key in channels:
    if key == 't':
      channels['t'] = channels['t'][::k]
    elif channels[key] is not None:
      if key in filtered:
        channels[key] = my_decimate (channels[key], k, method)
      else:
        channels[key] = channels[key][::k]

def find_gfm8_channels(rec):
  chan = {'Id':None,'Iq':None,'Vd':None,'Vq':None,'P':None,'Q':None,
          'Idc':None,'Vdc':None,'G':None,'T':None,'V0':None,'I0':None,
          'Md':None,'Mq':None,'Fc':None,'Dbar':None,'t':None}
  t = np.array(rec.time)
  chan['t'] = t
  for i in range(rec.analog_count):
    lbl = rec.analog_channel_ids[i]
    if 'V-node' in lbl:
      if 'DCP' in lbl:
        chan['Vdc'] = np.array (rec.analog[i])
      elif 'TEMP' in lbl:
        chan['T'] = np.array (rec.analog[i])
      elif 'FCTRL' in lbl:
        chan['Fc'] = np.array (rec.analog[i])
      elif 'MQ' in lbl:
        chan['Mq'] = np.array (rec.analog[i])
      elif 'MD' in lbl:
        chan['Md'] = np.array (rec.analog[i])
      elif 'G' in lbl:
        chan['G'] = np.array (rec.analog[i])
    elif 'I-branch' in lbl:
      if 'RSEQ' in lbl:
        chan['Idc'] = np.array (rec.analog[i])
      elif 'TACS   DBAR' in lbl:
        chan['Dbar'] = np.array (rec.analog[i])
      elif 'TACS   VCD' in lbl:
        chan['Vd'] = np.array (rec.analog[i])
      elif 'TACS   VCQ' in lbl:
        chan['Vq'] = np.array (rec.analog[i])
      elif 'TACS   VC0' in lbl:
        chan['V0'] = np.array (rec.analog[i])
      elif 'TACS   ICD' in lbl:
        chan['Id'] = np.array (rec.analog[i])
      elif 'TACS   ICQ' in lbl:
        chan['Iq'] = np.array (rec.analog[i])
      elif 'TACS   IC0' in lbl:
        chan['I0'] = np.array (rec.analog[i])
      elif 'TACS   PDQ0' in lbl:
        chan['P'] = np.array (rec.analog[i])
      elif 'TACS   QDQ0' in lbl:
        chan['Q'] = np.array (rec.analog[i])
  return chan
#  Data file [ GFM_v8.pl4]
#Type-4 entries (node voltages):
#   1 DCP      2 G        3 TEMP     4 FCTRL    5 MD       6 MQ
#Type-8 entries (branch voltages, * branch power):
#Type-9 entries (branch currents, * branch energy):
#  7  RSEQ  -DCP     8  TACS  -DBAR    9  TACS  -VCD    10  TACS  -VCQ
# 11  TACS  -ICD    12  TACS  -ICQ    13  TACS  -PDQ0   14  TACS  -QDQ0

def find_channels(rec):
  chan = {'Ia':None,'Ib':None,'Ic':None,'Va':None,'Vb':None,'Vc':None,'P':None,'Q':None,
        'Idc':None,'Vdc':None,'G':None,'T':None,
        'Iwa':None,'Iwb':None,'Iwc':None,'Vwa':None,'Vwb':None,'Vwc':None,
        'ImagA':None,'ImagB':None,'ImagC':None,'VmagA':None,'VmagB':None,'VmagC':None,
        'IangA':None,'IangB':None,'IangC':None,'VangA':None,'VangB':None,'VangC':None,
        'Fc':None,'F0':None,'F':None,'t':None}

  scaleRMS = 1.0 / math.sqrt(2.0)

# print('Analog Count', rec.analog_count)
# print('Status Count', rec.status_count)
# print('File Name', rec.filename)
# print('Station', rec.station_name)
# print('N', rec.total_samples)

  t = np.array(rec.time)
  chan['t'] = t
  for i in range(rec.analog_count):
    lbl = rec.analog_channel_ids[i]
  #  print (i, lbl)
    if 'V-node' in lbl:
      if 'PCCA' in lbl:
        chan['Vwa'] = np.array (rec.analog[i])
      elif 'PCCB' in lbl:
        chan['Vwb'] = np.array (rec.analog[i])
      elif 'PCCC' in lbl:
        chan['Vwc'] = np.array (rec.analog[i])
      elif 'VDC' in lbl:
        chan['Vdc'] = np.array (rec.analog[i])
    elif 'I-branch' in lbl:
      if 'PCCA' in lbl:
        chan['Iwa'] = np.array (rec.analog[i])
      elif 'PCCB' in lbl:
        chan['Iwb'] = np.array (rec.analog[i])
      elif 'PCCC' in lbl:
        chan['Iwc'] = np.array (rec.analog[i])
      elif 'DCND' in lbl:
        chan['Idc'] = np.array (rec.analog[i])
      elif 'MODELS PMEA' in lbl:
        chan['P'] = np.array (rec.analog[i])
      elif 'MODELS IRROUT' in lbl:
        chan['G'] = np.array (rec.analog[i])
      elif 'MODELS TMPOUT' in lbl:
        chan['T'] = np.array (rec.analog[i])
      elif 'MODELS VRMSA' in lbl:
        chan['Va'] = np.array (rec.analog[i])
      elif 'MODELS VRMSB' in lbl:
        chan['Vb'] = np.array (rec.analog[i])
      elif 'MODELS VRMSC' in lbl:
        chan['Vc'] = np.array (rec.analog[i])
      elif 'MODELS VMAGA' in lbl:
        chan['VmagA'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS VMAGB' in lbl:
        chan['VmagB'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS VMAGC' in lbl:
        chan['VmagC'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS VANGA' in lbl:
        chan['VangA'] = np.array (rec.analog[i])
      elif 'MODELS VANGB' in lbl:
        chan['VangB'] = np.array (rec.analog[i])
      elif 'MODELS VANGC' in lbl:
        chan['VangC'] = np.array (rec.analog[i])
      elif 'MODELS IRMSA' in lbl:
        chan['Ia'] = np.array (rec.analog[i])
      elif 'MODELS IRMSB' in lbl:
        chan['Ib'] = np.array (rec.analog[i])
      elif 'MODELS IRMSC' in lbl:
        chan['Ic'] = np.array (rec.analog[i])
      elif 'MODELS IMAGA' in lbl:
        chan['ImagA'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS IMAGB' in lbl:
        chan['ImagB'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS IMAGC' in lbl:
        chan['ImagC'] = scaleRMS * np.array (rec.analog[i])
      elif 'MODELS IANGA' in lbl:
        chan['IangA'] = np.array (rec.analog[i])
      elif 'MODELS IANGB' in lbl:
        chan['IangB'] = np.array (rec.analog[i])
      elif 'MODELS IANGC' in lbl:
        chan['IangC'] = np.array (rec.analog[i])
      elif 'MODELS QMEA' in lbl:
        chan['Q'] = np.array (rec.analog[i])
      elif 'MODELS FREQ' in lbl:
        chan['F0'] = np.array (rec.analog[i])
      elif 'MODELS FPLL' in lbl:
        chan['F'] = np.array (rec.analog[i])
      elif 'MODELS F' in lbl:
        chan['Fc'] = np.array (rec.analog[i])
      elif 'TACS   FREQA' in lbl:
        chan['F'] = np.array (rec.analog[i])
  return chan

def find_pv1_channels(rec): # for pv1_osg.atp
  chan = {'Idc':None,'Irms':None,'Vdc':None,'Vrms':None,'D':None,'Ppvpu':None,
        'G':None,'T':None,'Ud':None,'Uq':None,'Fc':None,
        'Vd':None, 'Vq':None, 'Id':None, 'Iq':None, 't':None}
  t = np.array(rec.time)
  chan['t'] = t
  for i in range(rec.analog_count):
    lbl = rec.analog_channel_ids[i]
    if 'V-node' in lbl:
      if 'TEMP' in lbl:
        chan['T'] = np.array (rec.analog[i])
      elif 'FCTRL' in lbl:
        chan['Fc'] = np.array (rec.analog[i])
      elif 'UD' in lbl:
        chan['Ud'] = np.array (rec.analog[i])
      elif 'UQ' in lbl:
        chan['Uq'] = np.array (rec.analog[i])
      elif 'G' in lbl:
        chan['G'] = np.array (rec.analog[i])
    elif 'V-branch' in lbl:
      if 'DCP' in lbl:
        chan['Vdc'] = np.array (rec.analog[i])
    elif 'I-branch' in lbl:
      if 'TACS' in lbl:
        if 'IRMS' in lbl:
          chan['Irms'] = np.array (rec.analog[i])
        elif 'VRMS' in lbl:
          chan['Vrms'] = np.array (rec.analog[i])
        elif 'FFREQ' in lbl:
          chan['F'] = np.array (rec.analog[i])
        elif 'DBAR' in lbl:
          chan['D'] = np.array (rec.analog[i])
        elif 'PPVPU' in lbl:
          chan['Ppvpu'] = np.array (rec.analog[i])
        elif 'VOD' in lbl:
          chan['Vd'] = np.array (rec.analog[i])
        elif 'VOQ' in lbl:
          chan['Vq'] = np.array (rec.analog[i])
        elif 'IOD' in lbl:
          chan['Id'] = np.array (rec.analog[i])
        elif 'IOQ' in lbl:
          chan['Iq'] = np.array (rec.analog[i])
      elif 'RSEQ' in lbl:
        chan['Idc'] = np.array (rec.analog[i])
  return chan

def find_lab1_channels(rec): # for pv1_osg.atp to mimic laboratory waveform data collection
  chan = {'Idc':None,'Iac':None,'Vdc':None,'Vac':None,'Ipcc':None,'Vpcc':None,
        'G':None,'T':None,'Ud':None,'Uq':None,'Fc':None,'Vp':None,'Ip':None,'Dv':None,'Wc':None,'Ang':None,
        'Vd':None, 'Vq':None, 'Id':None, 'Iq':None, 't':None}
  t = np.array(rec.time)
  chan['t'] = t
  for i in range(rec.analog_count):
    lbl = rec.analog_channel_ids[i]
    if 'V-node' in lbl:
      if 'TEMP' in lbl:
        chan['T'] = np.array (rec.analog[i])
      elif 'FCTRL' in lbl:
        chan['Fc'] = np.array (rec.analog[i])
      elif 'UD' in lbl:
        chan['Ud'] = np.array (rec.analog[i])
      elif 'UQ' in lbl:
        chan['Uq'] = np.array (rec.analog[i])
      elif 'G' in lbl:
        chan['G'] = np.array (rec.analog[i])
    elif 'V-branch' in lbl:
      if 'DCP    DCN' in lbl:
        chan['Vdc'] = np.array (rec.analog[i])
      elif 'LOAD   ACN' in lbl:
        chan['Vac'] = np.array (rec.analog[i])
    elif 'I-branch' in lbl:
      if 'TACS' in lbl:
        if 'DELTV' in lbl:
          chan['Dv'] = np.array (rec.analog[i])
        elif 'PCCV' in lbl:
          chan['Vpcc'] = np.array (rec.analog[i])
        elif 'PCCI' in lbl:
          chan['Ipcc'] = np.array (rec.analog[i])
        elif 'VOD' in lbl:
          chan['Vd'] = np.array (rec.analog[i])
        elif 'VOQ' in lbl:
          chan['Vq'] = np.array (rec.analog[i])
        elif 'IOD' in lbl:
          chan['Id'] = np.array (rec.analog[i])
        elif 'IOQ' in lbl:
          chan['Iq'] = np.array (rec.analog[i])
        elif 'INTWC' in lbl:
          chan['Ang'] = np.array (rec.analog[i])
        elif 'WC' in lbl:
          chan['Wc'] = np.array (rec.analog[i])
        elif 'VP' in lbl:
          chan['Vp'] = np.array (rec.analog[i])
        elif 'IP' in lbl:
          chan['Ip'] = np.array (rec.analog[i])
      elif 'RSEQ   DCP' in lbl:
        chan['Idc'] = np.array (rec.analog[i])
      elif 'ACP    LOAD' in lbl:
        chan['Iac'] = np.array (rec.analog[i])
  return chan

def load_atp_comtrade_channels (atp_base, filtered=None, k=1000, method='slice', pv1=False, lab1=False):
  rec = Comtrade ()
  rec.load (atp_base + '.cfg', atp_base + '.dat')
  if pv1:
    channels = find_pv1_channels (rec)
  elif lab1:
    channels = find_lab1_channels (rec)
  else:
    channels = find_gfm8_channels (rec)
  if 't' in channels: # since COMTRADE does not support sub-microsecond steps...
    channels['t'] = np.linspace (channels['t'][0], channels['t'][-1], len(channels['t']))
  if method is not None:
    decimate_channels (channels, k=k, method=method, filtered=filtered)
  return channels

def save_atp_channels (filename, groupname, channels, mode='w'):
  f = h5py.File (filename, mode)
  grp = f.create_group (groupname)
  for key, val in channels.items():
    if val is not None:
      grp.create_dataset (key, data=val, compression='gzip')
  f.close()

