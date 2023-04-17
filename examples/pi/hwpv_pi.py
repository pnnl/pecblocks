# Copyright (C) 2022-23 Battelle Memorial Institute
import json
import os
import sys
import h5py
import numpy as np
import math
import hwpv_evaluator as hwpv

Ctl_t = [-1.0, 2.50, 2.51, 200.0]
Ctl_y = [0.00, 0.00, 1.00, 1.00]

G_t = [-1.0, 1.00,   2.00,   200.0]
G_y = [0.00, 0.00, 1000.0,  1000.0]

T_t = [-1.0, 200.0]
T_y = [35.0, 35.0]

Md_t = [-1.00, 200.0]
Md_y = [0.810, 0.810]

Mq_t = [-1.00, 200.0]
Mq_y = [0.392, 0.392]

Fc_t = [-1.0, 200.0]
Fc_y = [60.0, 60.0]

R_t = [-1.0, 200.0]
R_y = [10.0, 10.0]

def control_input(x, y, t):
  return y

def evaluation_loop(cfg_filename, hdf5_filename, dt, tmax):
#  dt = 0.030
  n = int(tmax/dt) + 1

  fp = open (cfg_filename, 'r')
  cfg = json.load (fp)
  fp.close()

  mdl = hwpv.model ()
  mdl.set_sim_config (cfg)
  print ('Running {:s} model from {:s}'.format (mdl.name, cfg_filename))
  print ('  Input Channels: ', mdl.COL_U)
  print ('  Output Channels:', mdl.COL_Y)
  print ('  Training dt = {:.6f}s'.format (mdl.t_step))
  print ('  Running dt =  {:.6f}s to {:.4f}s for {:d} steps'.format (dt, tmax, n))
#  mdl.start_simulation_z ()
  mdl.start_simulation_s ()
  Id = 0.0
  Iq = 0.0
  vals = np.zeros((n,13)) # t, 8 inputs, 4 outputs
  i = 0
  while i < n:
    t = dt * i
    Ctl = np.interp (t, Ctl_t, Ctl_y)
    T = np.interp (t, T_t, T_y)
    G = np.interp (t, G_t, G_y)
    Md = np.interp (t, Md_t, Md_y)
    Mq = np.interp (t, Mq_t, Mq_y)
    Fc = np.interp (t, Fc_t, Fc_y)
    R = np.interp (t, R_t, R_y)
    Irms = math.sqrt(1.5) * math.sqrt(Id*Id + Iq*Iq)
    Vrms = Irms * R
    GVrms = G * Vrms
#    Vdc, Idc, Id, Iq = mdl.step_simulation_z (G=G, T=T, Md=Md, Mq=Mq, Fc=Fc, Vrms=Vrms, Ctl=Ctl, GVrms=GVrms)
    Vdc, Idc, Id, Iq = mdl.step_simulation_s (G=G, T=T, Md=Md, Mq=Mq, Fc=Fc, Vrms=Vrms, Ctl=Ctl, GVrms=GVrms, h=dt)
    vals[i,:] = [t, G, T, Md, Mq, Fc, Ctl, Vrms, GVrms, Vdc, Idc, Id, Iq]
    i += 1

  print ('Last Index', i, t, tmax)

  f = h5py.File (hdf5_filename, 'w')
  grp = f.create_group ('basecase')
  j = 0
  for key in ['t', 'G', 'T', 'Md', 'Mq', 'Fc', 'Ctl', 'Vrms', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq']:
    grp.create_dataset (key, data=vals[:,j], compression='gzip')
    j += 1
  f.close()

if __name__ == '__main__':
  cfg_filename = '../hwpv/big/balanced_fhf.json'
  hdf5_filename = 'hwpv_pi.hdf5'

  evaluation_loop (cfg_filename, hdf5_filename, dt=0.001, tmax=8.0)

