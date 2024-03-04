# Copyright (C) 2024 Battelle Memorial Institute
import json
import os
import sys
import pandas as pd
import pecblocks.pv3_poly as pv3_model
import math
import numpy as np
import matplotlib.pyplot as plt

cfg_filename = '../hwpv/ucf3/ucf3_fhf.json'
#cfg_filename = '../hwpv/ucf4/ucf4_fhf.json'
hdf5_filename = 'harness.hdf5'
tmax = 2.0

KRMS = math.sqrt(1.5)

G0 = 0.000023
G1 = 1000.0
def getG(t):
  return np.interp (t, [-1.0, 0.1, 1.1, 100.0], [G0, G0, G1, G1])

def getFc(t):
  return 60.0

def getT(t):
  return 35.0

def getCtl(t):
  return np.interp (t, [-1.0, 1.0, 1.010, 100.0], [0.0, 0.0, 1.0, 1.0])

R0 = 90.0
R1 = 65.0
def getRg(t):
  return np.interp (t, [-1.0, 6.0, 6.010, 100.0], [R0, R0, R1, R1])

MD0 = 0.999995
def getMd(t):
  return MD0

MQ0 = 0.001
def getMq(t):
  return MQ0

if __name__ == '__main__':

  fp = open (cfg_filename, 'r')
  cfg = json.load (fp)
  dt = cfg['t_step']
  fp.close()

  model = pv3_model.pv3 ()
  model.set_sim_config (cfg, model_only=False)
  model.start_simulation ()
  t = 0.0

  nsteps = 200 # for initialization of the model history terms
  Id = 0.0
  Iq = 0.0
  rows = []

#  print ('    Ts     Vd     Vq      G    GVrms     Md     Mq    Ctl    Vdc    Idc     Id     Iq')
  while t <= tmax:
    Rg = getRg(t)
    G = getG(t)
    Md = getMd(t)
    Mq = getMq(t)
    Fc = getFc(t)
    Ctl = getCtl(t)
    T = getT(t)
    Vd = Rg * Id
    Vq = Rg * Iq
    Vrms = KRMS * math.sqrt(Vd*Vd + Vq*Vq)
    GVrms = G * Vrms

    Vdc, Idc, Id, Iq = model.step_simulation (G=G, T=T, Md=Md, Mq=Mq, Fc=Fc, Vd=Vd, Vq=Vq, Ctl=Ctl, GVrms=GVrms, nsteps=nsteps)
    nsteps = 1

    Id = max(0.0, Id)
#    Iq = 0.0

#    print ('{:6.3f} {:6.2f} {:6.2f} {:6.1f} {:8.1f} {:6.3f} {:6.3f} {:6.1f} {:6.2f} {:6.3f} {:6.3f} {:6.3f}'.format(t, 
#            Vd, Vq, G, GVrms, Md, Mq, Ctl, Vdc, Idc, Id, Iq))
    dict = {'t':t,'G':G,'T':T,'Md':Md,'Mq':Mq,'Fc':Fc,'Ctl':Ctl,'Rg':Rg,'Vd':Vd,'Vq':Vq,'GVrms':GVrms,'Vdc':Vdc,'Idc':Idc,'Id':Id,'Iq':Iq}
    rows.append (dict)
    t += dt

  print ('simulation done, writing output to', hdf5_filename)
  df = pd.DataFrame (rows)
  df.to_hdf (hdf5_filename, key='basecase', mode='w', complevel=9)

  df.plot(x='t', y=['G', 'T', 'Md', 'Mq', 'Fc', 
                    'Ctl', 'Rg', 'Vd', 'Vq', 'GVrms', 
                    'Vdc', 'Idc', 'Id', 'Iq'], 
    layout=(3, 5), figsize=(15,8), subplots=True)
  plt.show()
