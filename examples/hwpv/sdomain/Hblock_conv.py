'''
D.Glover PNNL Power Electronics Team Summer 2022
HWPV Harold H-Block Conversion H(z) ---> H(s) from pv1_import.py
First block is test using input 1 to output 1 for all Harold conversion approximation methods, stability check
Second block converts all 9 H(z) tfs for each i/o case to dict using 'Forward Eulers' method
'''

import matplotlib.pyplot as plt
import numpy as np
import harold as har
print(har.__version__)
import control
import pickle
import os


# read in pickle file of H-block coeffs to dict from pv1.import.py
mypath = os.getcwd()
def readInHpickle(mypath):
  with open(os.path.join (mypath, 'H_coeffs.pkl'),'rb') as H_read: # read in .pkl
    dictData = pickle.load(H_read)
    print(dictData)
    H_read.close()
  return dictData


# convert Hz_sys_all to dictionary
def HzToDict(Hz_all):
  Hz_sys_all_dictionary = {}
  for i in range(0, 3): # inputs
    for j in range(0, 3): # outputs
      Hz_sys_all_dictionary.update({(i, j): Hz_all[i, j]})
  return Hz_sys_all_dictionary


def convertAllToHarold(a,b):
  mydict = {}
  for input in range(0,3):
    for output in range(0,3):
      H_z_har = har.Transfer(b[input][output],a[input][output],dt=dt)
      mydict.update({(input,output): H_z_har})
  return mydict # dict of all discrete time tf's in harold systems format {i/o : z-domain tf}


# Apply all H(z) --> H(s) conversion methods available
def applyUndiscretizeMethods(disc_methods, H_z):
  undisc_dict = {}
  for method in disc_methods:
    print("Converting with method:", method)
    try:
      H_s = har.undiscretize(H_z,method=method,prewarp_at=499.99,q='none') # tustin must be at half nyquist (1000Hz)
      undisc_dict.update({method:H_s})
    except ValueError as VE: # handles Zero-Order-Hold exception
      VE = 'The matrix logarithm returned a complex array, probably due to poles on the negative real axis,' \
          ' and a continous-time model cannot be obtained via this method without perturbations. '
  return undisc_dict


# convert cttf's from Harold back to Controls package
def haroldToControlsContTF(Hs_dict_harold):
  controls_Hs_dict = {}
  for method,Hs in Hs_dict_harold.items():
    numHs = np.array(Hs.num).ravel().tolist()
    denHs = np.array(Hs.den).ravel().tolist()
    Hs = control.TransferFunction(numHs, denHs)
    controls_Hs_dict.update({method: Hs})
  return controls_Hs_dict


# loop through dict and check poles for stability, return dict of stable cttfs ONLY
def checkPoles(Hs_dict):
  stable_dict = {}
  for method,cttf in Hs_dict.items():
    poles = control.pole(cttf)
    real_poles = np.real(poles)
    if np.all(real_poles < 0):
      stable_dict.update({method: cttf})
      # print(method + ' ' + 'method H(s) is stable')
    else:
      # print(method + ' ' + 'method H(s) is unstable')
      pass
  return stable_dict


# check all tf's in single PZMap plot
def checkStabilityPZMap(Cont_time_tfs_dict):
  for method,Hs in Cont_time_tfs_dict.items():
    control.pzmap(Hs,title='All Undiscretize Methods', plot=True, legend=False)
    # print(method,Hs)


# Check step, impulse responses of stable cttfs agains original Z-domain tf (H1_sys)
def getStep(Hz,Hs_dict):
  samples = 60001
  t = np.linspace(0,60,samples) # 60 sec check

  H1s_bil = Hs_dict.get('bilinear')
  H1s_tust = Hs_dict.get('tustin')
  H1s_for_diff = Hs_dict.get('forward difference')
  H1s_for_euler = Hs_dict.get('forward euler')
  H1s_for_rect = Hs_dict.get('forward rectangular')
  tbil,H1s_bil = control.step_response(H1s_bil,t,T_num=samples)
  ttust,H1s_tust = control.step_response(H1s_tust,t,T_num=samples)
  tdiff,H1s_for_diff = control.step_response(H1s_for_diff,t,T_num=samples)
  teuler,H1s_for_euler = control.step_response(H1s_for_euler,t,T_num=samples)
  trect,H1s_for_rect = control.step_response(H1s_for_rect,t,T_num=samples)
  tz,Hz = control.step_response(Hz,t,T_num=samples)

  plt.plot(tbil,H1s_bil, linestyle='solid', linewidth=1.75, alpha=0.7, color='g')
  plt.plot(ttust,H1s_tust, linestyle='dashed', linewidth=2, alpha=0.8, color='b')
  plt.plot(tdiff,H1s_for_diff, linestyle='solid', linewidth=1.5, alpha=0.6,color='c')
  plt.plot(teuler,H1s_for_euler, linestyle='dashed', linewidth=1.75, alpha=0.7, color='y')
  plt.plot(trect,H1s_for_rect, linestyle='dashdot', linewidth=1.75, alpha=0.8, color='r')
  plt.plot(tz, Hz, linestyle='dotted', linewidth=2, alpha=1, color='k')
  plt.xlabel('Time[s]')
  plt.ylabel('Magnitude')
  plt.legend(['H(s) Bilinear', 'H(s) Tustin', 'H(s) Forward Difference',
        'H(s) Forward Eulers', 'H(s) Forward Rectangular', 'H(Z) HWPV'])
  plt.grid()
  plt.title('Step Response HWPV Input 1 Output 1 H(z) to H(s)')
  plt.show()


def getImpulse(Hz,Hs_dict):
  samples = 60001
  t = np.linspace(0,60,samples)

  H1s_bil = Hs_dict.get('bilinear')
  H1s_tust = Hs_dict.get('tustin')
  H1s_for_diff = Hs_dict.get('forward difference')
  H1s_for_euler = Hs_dict.get('forward euler')
  H1s_for_rect = Hs_dict.get('forward rectangular')
  tbil,H1s_bil = control.impulse_response(H1s_bil,t,T_num=samples)
  ttust,H1s_tust = control.impulse_response(H1s_tust,t,T_num=samples)
  tdiff,H1s_for_diff = control.impulse_response(H1s_for_diff,t,T_num=samples)
  teuler,H1s_for_euler = control.impulse_response(H1s_for_euler,t,T_num=samples)
  trect,H1s_for_rect = control.impulse_response(H1s_for_rect,t,T_num=samples)
  tz,Hz = control.impulse_response(Hz,t,T_num=samples)

  plt.plot(tbil,H1s_bil, linestyle='solid', linewidth=1.75, alpha=0.7, color='g')
  plt.plot(ttust,H1s_tust, linestyle='dashed', linewidth=2, alpha=0.8, color='b')
  plt.plot(tdiff,H1s_for_diff, linestyle='solid', linewidth=1.5, alpha=0.6,color='c')
  plt.plot(teuler,H1s_for_euler, linestyle='dashed', linewidth=1.75, alpha=0.7, color='y')
  plt.plot(trect,H1s_for_rect, linestyle='dashdot', linewidth=1.75, alpha=0.8, color='r')
  plt.plot(tz, Hz, linestyle='dotted', linewidth=2, alpha=1, color='k')
  plt.xlabel('Time[s]')
  plt.ylabel('Magnitude')
  plt.legend(['H(s) Bilinear', 'H(s) Tustin', 'H(s) Forward Difference',
        'H(s) Forward Eulers', 'H(s) Forward Rectangular', 'H(Z) HWPV'])
  plt.grid()
  plt.title('Impulse Response HWPV Input 1 Output 1 H(z) to H(s)')
  plt.show()


# convert H(z)_sys_all_har to H(s) for each i/o
def convertToHsAll(Hz_dict_harold):
  Hs_dict_harold = {}
  for io,tf in Hz_dict_harold.items():
    Hs = har.undiscretize(tf,method='forward euler',prewarp_at=499.99,q='none')
    Hs_dict_harold.update({io:Hs})
  return Hs_dict_harold


# compare H(s) with H(z) step, 9 cases step and impulse 30secs
def plotStepResponse(Hz_sys,Hs_sys):
  samples = 30001
  t = np.linspace(0, 30, samples) # 30 secs simulation
  fig, ax = plt.subplots(3, 3, sharex='row', sharey='col', figsize=(3, 3))
  # axes = axes.ravel()
  row = 0
  col = 0
  for (ioHz,tfHz), (ioHs,tfHs) in zip(Hz_sys.items(), Hs_sys.items()):
    tz,Hzall = control.step_response(tfHz,t,T_num=samples)
    ts,Hsall = control.step_response(tfHs,t,T_num=samples)
    ax[row,col].plot(t,Hzall, linestyle='dotted', linewidth=3.25, alpha=1, color='k', label='H(z)')
    ax[row,col].plot(t,Hsall, linestyle='solid', linewidth=2.75, alpha=0.7, color='g', label='H(s)')
    ax[row,col].set_title(ioHs)
    ax[row,col].grid()
    ax[row,col].legend(loc="lower right")
    col +=1
    if col > 2:
      col = 0
      row +=1
    else: pass
    # print(row,col)
  # fig.legend(loc='lower right')
  plt.suptitle('HWPV MIMO H-Block Step Response (input,output) H(z) vs H(s)',fontsize=20)
  plt.show()


def plotImpulseResponse(Hz_sys,Hs_sys):
  samples = 30001
  t = np.linspace(0, 30, samples)
  fig, ax = plt.subplots(3, 3, sharex='row', sharey='col', figsize=(3, 3))
  # axes = axes.ravel()
  row = 0
  col = 0
  for (ioHz,tfHz), (ioHs,tfHs) in zip(Hz_sys.items(), Hs_sys.items()):
    tz,Hzall = control.impulse_response(tfHz,t,T_num=samples)
    ts,Hsall = control.impulse_response(tfHs,t,T_num=samples)
    ax[row,col].plot(t,Hzall, linestyle='dotted', linewidth=3.25, alpha=1, color='k', label='H(z)')
    ax[row,col].plot(t,Hsall, linestyle='solid', linewidth=2.75, alpha=0.7, color='r', label='H(s)')
    ax[row,col].set_title(ioHs)
    ax[row,col].grid()
    ax[row,col].legend(loc="lower right")
    col +=1
    if col > 2:
      col = 0
      row +=1
    else: pass
    # print(row,col)
  # fig.legend(loc='lower right')
  plt.suptitle('HWPV MIMO H-Block Impulse Response (input,output) H(z) vs H(s)',fontsize=20)
  plt.show()


# write H(s) to pickle (if needed)
def HsToPickle(mydict):
  import pickle
  with open("H(s)_tf.pkl", "wb") as Hs_write:
    pickle.dump(mydict, Hs_write)
    Hs_write.close()


def readInHofSpickle(path):
  with open(path + 'H(s)_tf.pkl','rb') as H_read:
    dictData = pickle.load(H_read)
    print(dictData)
    H_read.close()
  return dictData


# Sample Continuous H(s) tf's for equivalent Z-domain tf
def sampleHs(Hs_sys, sample_freq):
  Hz_dict = {}
  for key,val in Hs_sys.items():
    Hz_new = control.sample_system(val,sample_freq, 'euler')
    Hz_dict.update({key:Hz_new})
  return Hz_dict


# convert H1(z) to all negative powers for testing in PSCAD Hs_sys_all_stable VS Hz_sys_all_dict
def getNegZCoeffs(Hz_dict):
  import tbcontrol.conversion
  for key,tf in Hz_dict.items():
    H_neg_num, H_neg_den = tbcontrol.conversion.discrete_coeffs_pos_to_neg(tf.num[0][0], tf.den[0][0])
    print('G(Z) Num coeffs neg powers:', H_neg_num)
    print('G(Z) Den coeffs neg powers:',H_neg_den)




if __name__ == '__main__':
  dictData = readInHpickle(mypath)
  dt = 0.001  # time step (sample freq)
  a_coeffs, b_coeffs = dictData['a_array'], dictData['b_array']
  Hz_sys_all = control.TransferFunction(b_coeffs, a_coeffs, dt)  # all tf's in control form
  Hz_sys_all_dict = HzToDict(Hz_sys_all)  # holds dict of all original H(z) tf's

  ###################################################################################################################
  ###################################################################################################################
  ''' Input 1 to Output 1 Test'''
  # take only input 1 to output 1 tf for testing conv methods, stability
  H1z_sys = control.TransferFunction(b_coeffs[0][0], a_coeffs[0][0], dt)
  print("H1(z) Transfer Function:", H1z_sys)

  # conv H1(z) to Harold
  H1_z_har = har.Transfer(b_coeffs[0][0], a_coeffs[0][0], dt=dt)
  print("H1(z) Harold:", H1_z_har)

  # list all possible Harold "undiscretize" methods for conversion
  har_disc_methods = list(har._global_constants._KnownDiscretizationMethods)
  print(har_disc_methods)

  # dict with successful conversion methods to H1(s) continuous time tfs
  undisc_dict_H1z_har = applyUndiscretizeMethods(har_disc_methods, H1_z_har)
  # convert all H1(s) from Harold back to Python Controls library
  Hs1_ctrls_dict = haroldToControlsContTF(undisc_dict_H1z_har)

  # check poles of all H1(s) conv methods, return stable dict
  Hs1_ctrls_stable_dict = checkPoles(Hs1_ctrls_dict)

  ''' NOTE
  ** the biliear and tustin approximation methods produce identical H1(s) cttfs ** 
  ** the forward difference ,forward rectangular, and forward eulers produce identical H1(s) cttfs **
  '''

  # plot PZ-Map for stability check
  # checkStabilityPZMap(Hs1_ctrls_dict) # uncomment for PZ map (comment out getStep,getImpulse if used)

  ''' NOTE
  ** Bug in controls package opens pzmap with grid and legend from getStep and getImpulse **
  ** comment out to use either checkStabilityPZMap or getStep,getImpulse **
  '''

  # step, impulse response comparison for all stable H1(s) methods
  getStep(H1z_sys, Hs1_ctrls_stable_dict)
  getImpulse(H1z_sys, Hs1_ctrls_stable_dict)  # issue with initialization??

  ''' NOTE
  ** Step plot shows ONLY forward approximations are a match ** 
  '''

  ''' End Input 1 to Output 1 Test'''
  ###################################################################################################################
  ###################################################################################################################

  ###################################################################################################################
  ###################################################################################################################

  ''' From input 1 to output 1 test, 'Forward Eulers' method will be used for stable all H(z) ---> H(s)'''
  # get each tf for each i/o to Harold format
  Hz_sys_all_har = convertAllToHarold(a_coeffs, b_coeffs)

  # conv all H(z) to H(s) for each i/o and check stability using forward euler/difference approx
  Hs_sys_all_har = convertToHsAll(Hz_sys_all_har)
  Hs_sys_all_ctrls = haroldToControlsContTF(Hs_sys_all_har)
  Hs_sys_all_stable = checkPoles(Hs_sys_all_ctrls)

  # plot step, impulse responses, all 9 i/o cases
  plotStepResponse(Hz_sys_all_dict, Hs_sys_all_stable)
  plotImpulseResponse(Hz_sys_all_dict, Hs_sys_all_stable)

  # write H(s) dict to pickle file
  # HsToPickle(Hs_sys_all_stable)

  # create discrete time tf H(z) to compare with original from pv1_import.py
  Hz_one_msec = sampleHs(Hs_sys_all_stable, 1e-3)  # should be equivalent to original H(z)

  ''' End all conversions 'Forward Eulers' method stable H(z) ---> H(s)'''

  ###################################################################################################################
  ###################################################################################################################

  # PSCAD check both transfer functions input 1 to output 1
  # get from S-domain equivalent for PSCAD entry for input 1 to output 1 transfer function
  getNegZCoeffs(Hz_sys_all_dict)
  # show both tf's
  print("H1(z):", H1z_sys)
  print("H1(s) Coeffs:", Hs_sys_all_stable[0, 0])
