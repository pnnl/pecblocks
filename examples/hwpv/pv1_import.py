import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import control
import pv1_poly as pv1_model

model_folder = r'./'

Tmax = 4.0
dt = 0.001

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

aT=np.array([[  0.0, 30.0],
             [999.0, 30.0]])

aRG=np.array([[  0.0, 4.25],
              [999.0, 4.25]])

aFC=np.array([[  0.0, 60.0],
              [999.0, 60.0]])

aUD=np.array([[  0.0, 1.0],
              [999.0, 1.0]])

if __name__ == '__main__':
  model = pv1_model.pv1 ()
  model.load_sim_config (os.path.join(model_folder,'pv1_fhf_poly.json'))

  # make some arrays to hold plot data
  npts = int(Tmax/dt) + 1
  plt_t = np.zeros(npts)
  plt_g = np.zeros(npts)
  plt_ctl = np.zeros(npts)
  plt_rg = np.zeros(npts)
  plt_vdc = np.zeros(npts)
  plt_idc = np.zeros(npts)
  plt_irms = np.zeros(npts)
  plt_gvrms = np.zeros(npts)
  plt_vrms = np.zeros(npts)

  # simulation loop
  t = 0.0
  irms = 0.0 # need this to generate the first vrms from rg*irms
  for i in range(npts):
    # construct the inputs
    g = np.interp(t, aG[:,0], aG[:,1])
    rg = np.interp(t, aRG[:,0], aRG[:,1])
    ctl = np.interp(t, aCTL[:,0], aCTL[:,1])
    T = np.interp(t, aT[:,0], aT[:,1])
    ud = np.interp(t, aUD[:,0], aUD[:,1])
    fc = np.interp(t, aFC[:,0], aFC[:,1])
    vrms = rg * irms # lags by one time step
    gvrms = g * vrms

    # evaluate the HW model for outputs
    idc = 0.0
    vdc = 0.0
    irms = 0.0

    # save data for plotting (not necessary during simulation)
    plt_t[i] = t
    plt_g[i] = g
    plt_ctl[i] = ctl
    plt_rg[i] = rg
    plt_vdc[i] = vdc
    plt_idc[i] = idc
    plt_irms[i] = irms
    plt_gvrms[i] = gvrms
    plt_vrms[i] = vrms

    # advance the simulation time
    t += dt


  # plot the model's transfer function
  a_coeff = model.H1.a_coeff.detach().numpy()
  b_coeff = model.H1.b_coeff.detach().numpy()
  a_poly = np.empty_like(a_coeff, shape=(model.H1.out_channels, model.H1.in_channels, model.H1.n_a + 1))
  a_poly[:, :, 0] = 1
  a_poly[:, :, 1:] = a_coeff[:, :, :]
  b_poly = np.array(b_coeff)
  H1_sys = control.TransferFunction(b_poly, a_poly, dt)
  for i in range (model.H1.in_channels):
    for j in range (model.H1.out_channels):
      plt.figure()
      mag_H1, phase_H1, omega_H1 = control.bode(H1_sys[i, j])
      plt.suptitle('Transfer Function from Input {:d} to Output {:d}'.format (i, j))
#      plt.show()
      plt.savefig ('H1_{:d}_{:d}.png'.format (i, j))

  quit()

  fig, ax = plt.subplots (2, 4, sharex = 'col', figsize=(12,8), constrained_layout=True)
  fig.suptitle ('Simulating HWPV Model')
  ax[0,0].set_title ('G')
  ax[0,0].plot (plt_t, plt_g)
  ax[0,1].set_title ('Mode')
  ax[0,1].plot (plt_t, plt_ctl)
  ax[0,2].set_title ('Rgrid')
  ax[0,2].plot (plt_t, plt_rg)
  ax[0,3].set_title ('Vrms')
  ax[0,3].plot (plt_t, plt_vrms)
  ax[1,0].set_title ('G*Vrms')
  ax[1,0].plot (plt_t, plt_gvrms)
  ax[1,1].set_title ('Vdc')
  ax[1,1].plot (plt_t, plt_vdc)
  ax[1,2].set_title ('Idc')
  ax[1,2].plot (plt_t, plt_idc)
  ax[1,3].set_title ('Irms')
  ax[1,3].plot (plt_t, plt_irms)
  plt.show()

