# Copyright (C) 2022-2024 Battelle Memorial Institute
# this client just calculates and publishes Vrms=Rgrid*Irms
import json
import os
import sys
import helics
import time
import math

def newDouble(val, sub):
  if (sub is not None) and (helics.helicsInputIsUpdated(sub)):
    val = helics.helicsInputGetDouble(sub)
  return val

def newComplexMag(val, sub):
  if (sub is not None) and (helics.helicsInputIsUpdated(sub)):
    cval = helics.helicsInputGetComplex(sub)
    val = abs(cval)
  return val

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

  pub_Vd = None
  pub_Vq = None
  for i in range(pub_count):
    pub = helics.helicsFederateGetPublicationByIndex(h_fed, i)
    key = helics.helicsPublicationGetName(pub)
    print ('pub', i, key)
    if key.endswith('Vd'):
      pub_Vd = pub
    elif key.endswith('Vq'):
      pub_Vq = pub
    else:
      print (' ** could not match', key)

  sub_Id = None
  sub_Iq = None
  sub_Idc = None
  sub_Vdc = None
  sub_Ra = None
  sub_Rb = None
  sub_Rc = None
  for i in range(sub_count):
    sub = helics.helicsFederateGetInputByIndex(h_fed, i)
    key = helics.helicsInputGetTarget(sub)
    print ('sub', i, key)
    if key.endswith('Id'):
      sub_Id = sub
    elif key.endswith('Iq'):
      sub_Iq = sub
    elif key.endswith('Idc'):
      sub_Idc = sub
    elif key.endswith('Vdc'):
      sub_Vdc = sub
    elif key.endswith('Ra'):
      sub_Ra = sub
    elif key.endswith('Rb'):
      sub_Rb = sub
    elif key.endswith('Rc'):
      sub_Rc = sub
    else:
      print (' ** could not match', key)

  Id = 0.0 # 0+0j
  Iq = 0.0
  Idc = 0.0
  Vdc = 0.0
  Ra = 0.0
  Rb = 0.0
  Rc = 0.0
  Rg = 0.0
  ts = 0

  helics.helicsFederateEnterExecutingMode(h_fed)
  while ts < tmax:
    Ra = newDouble (Ra, sub_Ra)
    Rb = newDouble (Rb, sub_Rb)
    Rc = newDouble (Rc, sub_Rc)
    Vdc = newDouble (Vdc, sub_Vdc)
    Idc = newDouble (Idc, sub_Idc)
    Id = newComplexMag (Id, sub_Id)
    Iq = newComplexMag (Iq, sub_Iq)
    Rg = (Ra + Rb + Rc) / 3.0
    Vd = Rg * Id
    Vq = Rg * Iq
#    print ('{:6.3f}, Id={:.3f}, Iq={:.3f}, Rg={:.3f}, Vd={:.3f}, Vq={:.3f}'.format(ts, Id, Iq, Rg, Vd, Vq))
    if pub_Vd is not None:
      helics.helicsPublicationPublishComplex(pub_Vd, Vd+0j)
    if pub_Vq is not None:
      helics.helicsPublicationPublishComplex(pub_Vq, Vq+0j)
    ts = helics.helicsFederateRequestTime(h_fed, tmax)
  helics.helicsFederateDestroy(h_fed)

if __name__ == '__main__':
  t0 = time.process_time()
  cfg_filename = 'pv3_client.json'
  helics_loop(cfg_filename)
  t1 = time.process_time()
  print ('PV3 Client elapsed time = {:.4f} seconds.'.format (t1-t0))

