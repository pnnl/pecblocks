# Copyright (C) 2022 Battelle Memorial Institute
# this client just calculates and publishes Vrms=Rgrid*Irms
import json
import os
import sys
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
  for i in range(pub_count):
    pub = helics.helicsFederateGetPublicationByIndex(h_fed, i)
    key = helics.helicsPublicationGetName(pub)
    print ('pub', i, key)
    if key.endswith('Vrms'):
      pub_Vrms = pub
    else:
      print (' ** could not match', key)

  sub_Vs = None
  sub_Is = None
  sub_Ic = None
  sub_idc = None
  sub_vdc = None
  sub_Rg = None
  for i in range(sub_count):
    sub = helics.helicsFederateGetInputByIndex(h_fed, i)
    key = helics.helicsInputGetTarget(sub)
    print ('sub', i, key)
    if key.endswith('Vs'):
      sub_Vs = sub
    elif key.endswith('Is'):
      sub_Is = sub
    elif key.endswith('Ic'):
      sub_Ic = sub
    elif key.endswith('idc'):
      sub_idc = sub
    elif key.endswith('vdc'):
      sub_vdc = sub
    elif key.endswith('Rg'):
      sub_Rg = sub
    else:
      print (' ** could not match', key)

  print (sub_Vs)
  print (sub_Is)
  print (sub_Ic)
  print (sub_idc)
  print (sub_vdc)
  print (sub_Rg)

  Vs = 0+0j
  Is = 0+0j
  Ic = 0+0j
  idc = 0.0
  vdc = 0.0
  Rg = 0.0
  ts = 0

  print ('    ts       Vs       Is       Ic       Rg')
  helics.helicsFederateEnterExecutingMode(h_fed)
  while ts < tmax:
    if (sub_Vs is not None) and (helics.helicsInputIsUpdated(sub_Vs)):
      Vs = helics.helicsInputGetComplex(sub_Vs)
#      print ('  new Vs')
    if (sub_Is is not None) and (helics.helicsInputIsUpdated(sub_Is)):
      Is = helics.helicsInputGetComplex(sub_Is)
#      print ('  new Is')
    if (sub_Ic is not None) and (helics.helicsInputIsUpdated(sub_Ic)):
      Ic = helics.helicsInputGetComplex(sub_Ic)
#      print ('  new Ic')
    if (sub_idc is not None) and (helics.helicsInputIsUpdated(sub_idc)):
      idc = helics.helicsInputGetDouble(sub_idc)
#      print ('  new idc')
    if (sub_vdc is not None) and (helics.helicsInputIsUpdated(sub_vdc)):
      vdc = helics.helicsInputGetDouble(sub_vdc)
#      print ('  new vdc')
    if (sub_Rg is not None) and (helics.helicsInputIsUpdated(sub_Rg)):
      Rg = helics.helicsInputGetDouble(sub_Rg)
#      print ('  new Rg')
    print ('{:6.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}'.format(ts, abs(Vs), abs(Is), abs(Ic), Rg))
    Vc = Rg * Ic
    if pub_Vrms is not None:
      helics.helicsPublicationPublishComplex(pub_Vrms, Vc)
    ts = helics.helicsFederateRequestTime(h_fed, tmax)
  helics.helicsFederateDestroy(h_fed)

if __name__ == '__main__':
  t0 = time.process_time()
  cfg_filename = 'pv1_client.json'
  helics_loop(cfg_filename)
  t1 = time.process_time()
  print ('PV1 Client elapsed time = {:.4f} seconds.'.format (t1-t0))

