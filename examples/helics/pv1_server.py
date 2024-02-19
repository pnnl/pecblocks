# Copyright (C) 2022-2024 Battelle Memorial Institute
import json
import os
import sys
import pandas as pd
import helics
import time
import pecblocks.pv1_poly as pv1_model
#import h5py

def helics_loop(cfg_filename, hdf5_filename):
  fp = open (cfg_filename, 'r')
  cfg = json.load (fp)
  tmax = cfg['application']['Tmax']
  fp.close()

  model = pv1_model.pv1 ()
  model.set_sim_config (cfg['application'], model_only=False)
  Lf = 2.0   # mH
  Cf = 20.0  # uH
  Lc = 0.4   # mH
  model.set_LCL_filter (Lf=Lf*1.0e-3, Cf=Cf*1.0e-6, Lc=Lc*1.0e-3)
  model.start_simulation ()

  h_fed = helics.helicsCreateValueFederateFromConfig(cfg_filename)
  fed_name = helics.helicsFederateGetName(h_fed)
  pub_count = helics.helicsFederateGetPublicationCount(h_fed)
  sub_count = helics.helicsFederateGetInputCount(h_fed)
  period = int(helics.helicsFederateGetTimeProperty(h_fed, helics.helics_property_time_period))
  print('Federate {:s} has {:d} pub and {:d} sub, {:d} period'.format(fed_name, pub_count, sub_count, period), flush=True)

  pub_vdc = None
  pub_idc = None
  pub_Vs = None
  pub_Is = None
  pub_Ic = None
  for i in range(pub_count):
    pub = helics.helicsFederateGetPublicationByIndex(h_fed, i)
    key = helics.helicsPublicationGetName(pub)
    print ('pub', i, key)
    if key.endswith('vdc'):
      pub_vdc = pub
    elif key.endswith('idc'):
      pub_idc = pub
    elif key.endswith('Vs'):
      pub_Vs = pub
    elif key.endswith('Is'):
      pub_Is = pub
    elif key.endswith('Ic'):
      pub_Ic = pub
    else:
      print (' ** could not match', key)

  sub_Vrms = None
  sub_G = None
  sub_T = None
  sub_Ud = None
  sub_Fc = None
  sub_ctl = None
  for i in range(sub_count):
    sub = helics.helicsFederateGetInputByIndex(h_fed, i)
    key = helics.helicsInputGetTarget(sub)
    print ('sub', i, key)
    if key.endswith('Vrms'):
      sub_Vrms = sub
    elif key.endswith('G'):
      sub_G = sub
    elif key.endswith('T'):
      sub_T = sub
    elif key.endswith('Ud'):
      sub_Ud = sub
    elif key.endswith('Fc'):
      sub_Fc = sub
    elif key.endswith('ctl'):
      sub_ctl = sub
    else:
      print (' ** could not match', key)

  Vc = 0+0j
  T = 0.0
  G = 0.0
  Ud = 0.0
  Fc = 0.0
  ctl = 0.0
  ts = 0
  nsteps = 1000 # for initialization of the model history terms, dt=1ms
  rows = []
  print (sub_G)
  print (sub_T)
  print (sub_Ud)
  print (sub_Fc)
  print (sub_ctl)

  helics.helicsFederateEnterExecutingMode(h_fed)
  # some notes on helicsInput timing
  #  1) initial values are garbage until the other federate actually publishes
  #  2) helicsInputIsValid checks the subscription pipeline for validity, but not the value
  #  3) helicsInputIsUpdated resets to False immediately after you read the value,
  #     will become True if value changes later
  #  4) helicsInputLastUpdateTime is > 0 only after the other federate published its first value
  while ts < tmax:
    if (sub_ctl is not None) and (helics.helicsInputIsUpdated(sub_ctl)):
      ctl = helics.helicsInputGetDouble(sub_ctl)
      print ('  new ctl')
    if (sub_T is not None) and (helics.helicsInputIsUpdated(sub_T)):
      T = helics.helicsInputGetDouble(sub_T)
      print ('  new T')
    if (sub_Ud is not None) and (helics.helicsInputIsUpdated(sub_Ud)):
      Ud = helics.helicsInputGetDouble(sub_Ud)
      print ('  new Ud')
    if (sub_Fc is not None) and (helics.helicsInputIsUpdated(sub_Fc)):
      Fc = helics.helicsInputGetDouble(sub_Fc)
      print ('  new Fc', ts, Fc)
    if (sub_Vrms is not None) and (helics.helicsInputIsUpdated(sub_Vrms)):
      Vc = helics.helicsInputGetComplex(sub_Vrms)
      print ('  new Vc')
    if (sub_G is not None) and (helics.helicsInputIsUpdated(sub_G)):
      G = helics.helicsInputGetDouble(sub_G)
      print ('  new G')
    Vrms = abs(Vc)
    GVrms = 0.001 * G * Vrms
    print ('{:6.3f}, Vrms={:.3f}, G={:.1f}, GVrms={:.3f}, T={:.3f}, Ud={:.3f}, Fc={:.3f}'.format(ts, Vrms, G, GVrms, T, Ud, Fc))
    vdc, idc, irms, Vs, Is = model.step_simulation (G=G, T=T, Ud=Ud, Fc=Fc, Vrms=Vrms, Mode=ctl, GVrms=GVrms, nsteps=nsteps)
    nsteps = 1
    Ic = irms+0j
    if pub_idc is not None:
      helics.helicsPublicationPublishDouble(pub_idc, idc)
    if pub_vdc is not None:
      helics.helicsPublicationPublishDouble(pub_vdc, vdc)
    if pub_Vs is not None:
      helics.helicsPublicationPublishComplex(pub_Ic, Ic)
    if pub_Is is not None:
      helics.helicsPublicationPublishComplex(pub_Is, Is)
    if pub_Vs is not None:
      helics.helicsPublicationPublishComplex(pub_Vs, Vs)
    dict = {'t':ts,'G':G,'T':T,'Ud':Ud,'Fc':Fc,'Ctl':ctl,'Vc':Vc,'Vs':Vs,'Ic':Ic,'Is':Is,'Vdc':vdc,'Idc':idc}
    rows.append (dict)
    ts = helics.helicsFederateRequestTime(h_fed, tmax)
  helics.helicsFederateDestroy(h_fed)

  print ('simulation done, writing output to', hdf5_filename)
  df = pd.DataFrame (rows)
  df.to_hdf (hdf5_filename, 'basecase', mode='w', complevel=9)

if __name__ == '__main__':
  t0 = time.process_time()
  cfg_filename = 'pv1_server.json'
  hdf5_filename = 'pv1_server.hdf5'
  helics_loop(cfg_filename, hdf5_filename)
  t1 = time.process_time()
  print ('PV1 Server elapsed time = {:.4f} seconds.'.format (t1-t0))

