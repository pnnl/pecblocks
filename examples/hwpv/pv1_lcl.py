import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import control
import pv1_poly as pv1_model
import time
import h5py
import dynonet.metrics

model_folder = r'./models'

Tmax = 8.000
dt = 0.001

do_atp = True

# testing data
aCTL=np.array([[  0.0,  0.0],
               [  2.0,  0.0],
               [  2.1,  1.0],
               [999.0,  1.0]])

aG=np.array([[  0.0,   0.0],
             [  0.1,   0.0],
             [  0.2, 950.0],
             [  2.4, 950.0],
             [  2.5, 825.0],
             [999.0, 825.0]])

aT=np.array([[  0.0,  5.0],
             [  3.4,  5.0],
             [  3.5, 35.0],
             [999.0, 35.0]])

aFC=np.array([[  0.0, 60.0],
              [  4.4, 60.0],
              [  4.5, 63.0],
              [999.0, 63.0]])

aUD=np.array([[  0.0, 1.00],
              [  5.4, 1.00],
              [  5.5, 0.92],
              [999.0, 0.92]])

aRG=np.array([[  0.0, 4.25],
              [  6.4, 4.25],
              [  6.5, 6.25],
              [999.0, 6.25]])

if __name__ == '__main__':
  model = pv1_model.pv1 ()
  model.load_sim_config (os.path.join(model_folder,'pv1_fhf_poly.json'), model_only=False)

  # make some arrays to hold plot data
  npts = int(Tmax/dt) + 1
  plt_t = np.zeros(npts)
  plt_vc_mag = np.zeros(npts)
  plt_ic_mag = np.zeros(npts)
  plt_vc_ang = np.zeros(npts)
  plt_ic_ang = np.zeros(npts)
  plt_vs_mag = np.zeros(npts)
  plt_is_mag = np.zeros(npts)
  plt_vs_ang = np.zeros(npts)
  plt_is_ang = np.zeros(npts)

  # simulation loop
  t0 = time.process_time()
  t = 0.0
  irms = 0.0 # need this to generate the first vrms from rg*irms
  Lf = 2.0   # mH
  Cc = 20.0  # uH
  Lc = 0.4   # mH
  model.set_LCL_filter (Lf=Lf*1.0e-3, Cc=Cc*1.0e-6, Lc=Lc*1.0e-3)
  model.start_simulation ()
  for i in range(npts):
    # construct the inputs
    g = np.interp(t, aG[:,0], aG[:,1])
    rg = np.interp(t, aRG[:,0], aRG[:,1])
    ctl = np.interp(t, aCTL[:,0], aCTL[:,1])
    T = np.interp(t, aT[:,0], aT[:,1])
    ud = np.interp(t, aUD[:,0], aUD[:,1])
    fc = np.interp(t, aFC[:,0], aFC[:,1])
    vrms = rg * irms # lags by one time step
    gvrms = 0.001 * g * vrms

    # evaluate the HW model for outputs
    vdc, idc, irms, Vs, Is = model.step_simulation (G=g, T=T, Ud=ud, Fc=fc, Vrms=vrms, 
                                            Mode=ctl, GVrms=gvrms)

    # save data for plotting (not necessary during simulation)
    plt_t[i] = t
    plt_vc_mag[i] = np.abs(vrms)
    plt_ic_mag[i] = np.abs(irms)
    if plt_vc_mag[i] > 0.0:
      plt_vc_ang[i] = np.angle (vrms, deg=True)
    if plt_ic_mag[i] > 0.0:
     plt_ic_ang[i] = np.angle (irms, deg=True)
    plt_vs_mag[i] = np.abs(Vs)
    plt_is_mag[i] = np.abs(Is)
    if plt_vs_mag[i] > 0.0:
      plt_vs_ang[i] = np.angle (Vs, deg=True)
    if plt_is_mag[i] > 0.0:
      plt_is_ang[i] = np.angle (Is, deg=True)

    # advance the simulation time
    t += dt

  t1 = time.process_time()
  print ('Simulation elapsed time = {:.4f} seconds.'.format (t1-t0))

  if do_atp:
    h5file = 'c:/src/atptools/pv1_iir.hdf5'
    print ('reading ATP data from', h5file)
    with h5py.File(h5file, 'r') as f:
      for grp_name, grp in f.items():
        dlen = grp['t'].len()
        print (grp_name, 'has', dlen, 'points')
        atp_t = np.zeros(dlen)
        atp_irms = np.zeros(dlen)
        atp_vrms = np.zeros(dlen)
        grp['t'].read_direct (atp_t)
        grp['Irms'].read_direct (atp_irms)
        grp['Vrms'].read_direct (atp_vrms)

  fig, ax = plt.subplots (2, 3, sharex = 'col', figsize=(12,8), constrained_layout=True)
  fig.suptitle ('Simulating HWPV Model with LCL Filters; Lf = {:.2f} mH, Lc = {:.2f} mH, Cc = {:.2f} uF'.format(Lf, Lc, Cc))

  ax[0,0].set_title ('Voltage Magnitudes')
  if do_atp:
    ax[0,0].plot (atp_t, atp_vrms, 'b', label='ATP Vc')
  ax[0,0].plot (plt_t, plt_vc_mag, 'r', label='IIR Vc')
  ax[0,0].plot (plt_t, plt_vs_mag, 'g', label='IIR Vs')
  ax[0,0].legend(loc='best')

  ax[0,1].set_title ('Vs - Vc (IIR)')
  ax[0,1].plot (plt_t, plt_vs_mag - plt_vc_mag, 'g')

  ax[0,2].set_title ('Voltage Angles [deg]')
  ax[0,2].plot (plt_t, plt_vc_ang, 'r', label='IIR Vc')
  ax[0,2].plot (plt_t, plt_vs_ang, 'g', label='IIR Vs')
  ax[0,2].legend(loc='best')

  ax[1,0].set_title ('Current Magnitudes')
  if do_atp:
    ax[1,0].plot (atp_t, atp_irms, 'b', label='ATP Ic')
  ax[1,0].plot (plt_t, plt_ic_mag, 'r', label='IIR Ic')
  ax[1,0].plot (plt_t, plt_is_mag, 'g', label='IIR Is')
  ax[1,0].legend(loc='best')

  ax[1,1].set_title ('Is - Ic (IIR)')
  ax[1,1].plot (plt_t, plt_is_mag - plt_ic_mag, 'g')

  ax[1,2].set_title ('Current Angles [deg]')
  ax[1,2].plot (plt_t, plt_ic_ang, 'r', label='IIR Ic')
  ax[1,2].plot (plt_t, plt_is_ang, 'g', label='IIR Is')
  ax[1,2].legend(loc='best')

  for row in range(2):
    for col in range(3):
      ax[row,col].grid()
  plt.show()

