# Copyright (C) 2022 Battelle Memorial Institute
import json
import os
import sys
# import numpy as np
import helics
import time

def helics_loop(cfg_filename):
  fp = open (cfg_filename, 'r')
  cfg = json.load (fp)
  tmax = cfg['application']['Tmax']
  fp.close()

  h_fed = helics.helicsCreateValueFederateFromConfig(cfg_filename)
  fed_name = helics.helicsFederateGetName(h_fed)
  pub_count = helics.helicsFederateGetPublicationCount(h_fed)
  sub_count = helics.helicsFederateGetInputCount(h_fed)
  period = int(helics.helicsFederateGetTimeProperty(h_fed, helics.helics_property_time_period))
  print('Federate {:s} has {:d} pub and {:d} sub, {:d} period'.format(fed_name, pub_count, sub_count, period), flush=True)

  pub_Vrms = None
  pub_GVrms = None
  for i in range(pub_count):
    pub = helics.helicsFederateGetPublicationByIndex(h_fed, i)
    key = helics.helicsPublicationGetName(pub)
    print ('pub', i, key)
    if 'GVrms' in key:
      pub_GVrms = pub
    elif 'Vrms' in key:
      pub_Vrms = pub
    else:
      print (' ** could not match', key)

  sub_Vs = None
  sub_Is = None
  sub_Ic = None
  sub_idc = None
  sub_vdc = None
  sub_Rg = None
  sub_G = None
  for i in range(sub_count):
    sub = helics.helicsFederateGetInputByIndex(h_fed, i)
    key = helics.helicsSubscriptionGetTarget(sub)
    print ('sub', i, key)
    if 'Vs' in key:
      sub_Vs = sub
    elif 'Is' in key:
      sub_Is = sub
    elif 'Ic' in key:
      sub_Ic = sub
    elif 'G' in key:
      sub_G = sub
    elif 'idc' in key:
      sub_idc = sub
    elif 'vdc' in key:
      sub_vdc = sub
    elif 'Rg' in key:
      sub_Rg = sub
    else:
      print (' ** could not match', key)

  Vs = 0+0j
  Is = 0+0j
  Ic = 0+0j
  idc = 0.0
  vdc = 0.0
  Rg = 0.0
  G = 0.0
  ts = 0

  helics.helicsFederateEnterExecutingMode(h_fed)
  while ts < tmax:
    if (sub_Vs is not None) and (helics.helicsInputIsUpdated(sub_Vs)):
      Vs = helics.helicsInputGetComplex(sub_Vs)
    if (sub_Is is not None) and (helics.helicsInputIsUpdated(sub_Is)):
      Is = helics.helicsInputGetComplex(sub_Is)
    if (sub_Ic is not None) and (helics.helicsInputIsUpdated(sub_Ic)):
      Ic = helics.helicsInputGetComplex(sub_Ic)
    if (sub_G is not None) and (helics.helicsInputIsUpdated(sub_G)):
      G = helics.helicsInputGetDouble(sub_G)
    if (sub_idc is not None) and (helics.helicsInputIsUpdated(sub_idc)):
      idc = helics.helicsInputGetDouble(sub_idc)
    if (sub_vdc is not None) and (helics.helicsInputIsUpdated(sub_vdc)):
      vdc = helics.helicsInputGetDouble(sub_vdc)
    if (sub_Rg is not None) and (helics.helicsInputIsUpdated(sub_Rg)):
      Rg = helics.helicsInputGetDouble(sub_Rg)
    print ('{:6.3f}, Vs={:.3f}, Is={:.3f}, Ic={:.3f}, Rg={:.3f}, G={:.3f}'.format(ts, abs(Vs), abs(Is), abs(Ic), Rg, G))
    ts = helics.helicsFederateRequestTime(h_fed, tmax)
  helics.helicsFederateDestroy(h_fed)

if __name__ == '__main__':
  t0 = time.process_time()
  cfg_filename = 'pv1_client.json'
  helics_loop(cfg_filename)
  t1 = time.process_time()
  print ('PV1 Client elapsed time = {:.4f} seconds.'.format (t1-t0))

