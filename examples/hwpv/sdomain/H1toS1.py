import matplotlib.pyplot as plt
import numpy as np
import harold as har
import control
import pickle
import os
import torch

mypkl = 'H1.pkl'
mypath = os.getcwd()
myfile = os.path.join (mypath, mypkl)

def readInHpickle():
  block = torch.load(myfile)
  a = block['a_coeff'].numpy().squeeze()
  b = block['b_coeff'].numpy().squeeze()
  return a, b

# convert Hz_sys_all to dictionary
def HzToDict(Hz_all, n_in, n_out):
  Hz_sys_all_dictionary = {}
  for i in range(n_in): # inputs
    for j in range(n_out): # outputs
      Hz_sys_all_dictionary.update({(i, j): Hz_all[i, j]})
  return Hz_sys_all_dictionary

def convertAllToHarold(a, b, n_in, n_out):
  mydict = {}
  for input in range(n_in):
    for output in range(n_out):
      H_z_har = har.Transfer(b[input][output],a[input][output],dt=dt)
      mydict.update({(input,output): H_z_har})
  return mydict # dict of all discrete time tf's in harold systems format {i/o : z-domain tf}

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
#    print ('Method', method)
#    print ('CTTF', cttf)
    poles = control.pole(cttf)
#    print (poles)
    real_poles = np.real(poles)
    print (method, real_poles)
    if np.all(real_poles < 0):
      stable_dict.update({method: cttf})
      # print(method + ' ' + 'method H(s) is stable')
    else:
      print ('unstable', cttf)
#      print(method + ' ' + 'method H(s) is unstable')
      pass
  return stable_dict

# check all tf's in single PZMap plot
def checkStabilityPZMap(Cont_time_tfs_dict):
  for method,Hs in Cont_time_tfs_dict.items():
    control.pzmap(Hs,title='All Undiscretize Methods', plot=True, legend=False)
    # print(method,Hs)

# convert H(z)_sys_all_har to H(s) for each i/o
def convertToHsAll(Hz_dict_harold):
  Hs_dict_harold = {}
  for io,tf in Hz_dict_harold.items():
    Hs = har.undiscretize(tf,method='forward euler',prewarp_at=499.99,q='none')
    Hs_dict_harold.update({io:Hs})
  return Hs_dict_harold

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
  dt = 0.002  # time step (sample freq)
  a_coeffs, b_coeffs = readInHpickle()
  print ('Read A and B from PKL file, shapes', a_coeffs.shape, b_coeffs.shape)
  n_b = b_coeffs.shape[0]
  n_a = a_coeffs.shape[0]
  n_in = a_coeffs.shape[1]
  n_out = a_coeffs.shape[2]
  print ('n_a={:d}, n_b={:d}, n_in={:d}, n_out={:d}'.format (n_a, n_b, n_in, n_out))
  Hz_sys_all = control.TransferFunction(b_coeffs, a_coeffs, dt)  # all tf's in control form
  Hz_sys_all_dict = HzToDict(Hz_sys_all, n_in, n_out)  # holds dict of all original H(z) tf's

  # get each tf for each i/o to Harold format
  Hz_sys_all_har = convertAllToHarold(a_coeffs, b_coeffs, n_in, n_out)

  # conv all H(z) to H(s) for each i/o and check stability using forward euler/difference approx
  Hs_sys_all_har = convertToHsAll(Hz_sys_all_har)
  Hs_sys_all_ctrls = haroldToControlsContTF(Hs_sys_all_har)
  Hs_sys_all_stable = checkPoles(Hs_sys_all_ctrls)

  # create discrete time tf H(z) to compare with original
  Hz_one_msec = sampleHs(Hs_sys_all_stable, dt)  # should be equivalent to original H(z)

  # get from S-domain equivalent for PSCAD entry for input 1 to output 1 transfer function
#  getNegZCoeffs(Hz_sys_all_dict)
  # show both tf's
  print('H1(z):', Hz_sys_all[0, 0])
  print('H1(s) Coeffs:', Hs_sys_all_stable[0, 0])
  print('Sampled', Hz_one_msec[0,0])

