# copyright 2021-2025 Battelle Memorial Institute

"""
  Functions to perform sensitivity analysis on Thevenin and Norton controlled
  sources containing generalized block diagram models.
"""

import numpy as np
import pecblocks.pv3_poly as pv3_model
import math
import torch

KRMS = math.sqrt(1.5)

minRd = 1.0e9 
"""Minimum resistance *Rd* found in a model's training dataset"""
maxRd = 0.0
"""Maximum resistance *Rd* found in a model's training dataset"""
minRq = 1.0e9
"""Minimum resistance *Rq* found in a model's training dataset"""
maxRq = 0.0
"""Maximum resistance *Rq* found in a model's training dataset"""

nRd = 0 
"""Number of *Rd* changes found in a model's training dataset"""
nG = 0
"""Number of *G* changes found in a model's training dataset"""
nUd = 0
"""Number of *Ud* changes found in a model's training dataset"""
nUq = 0
"""Number of *Uq* changes found in a model's training dataset"""

def analyze_case(model, idx, bPrint=False):
  """Scans a case to update the ranges of grid resistance and parameter changes in a model's training dataset. *Internal*

  Args:
    model (pv3_poly): The *pv3_poly* instance with data loaded and normalized.
    idx (int): the zero-based case number to scan.
    bPrint (bool): request diagnostic output of step changes identified in this case.

  Yields:
    Updates to global module variables *minRd*, *maxRd*, *minRq*, *maxRq*, *nRd*, *nG*, *nUd*, and *nUq*.

  Returns:
    None
  """
  global minRd, maxRd, minRq, maxRq
  global nRd, nG, nT, nFc, nUd, nUq
  i1 = 180 # 990
  i2 = 380 # 2499

  rmse, mae, y_hat, y_true, u = model.testOneCase(idx, npad=100)

  jId = model.COL_Y.index ('Id')
  jIq = model.COL_Y.index ('Iq')
  jVd = model.COL_U.index ('Vd')
  jVq = model.COL_U.index ('Vq')

  jG = model.COL_U.index ('G')
  jUd = model.COL_U.index ('Ud')
  jUq = model.COL_U.index ('Uq')

  dG = (u[i2,jG] - u[i1,jG]) * model.normfacs['G']['scale']
  dUd = (u[i2,jUd] - u[i1,jUd]) * model.normfacs['Ud']['scale']
  dUq = (u[i2,jUq] - u[i1,jUq]) * model.normfacs['Uq']['scale']

  Rd1 = abs ((u[i1,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']) / (y_true[i1,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  Rd2 = abs ((u[i2,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']) / (y_true[i2,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']))
  dRd = Rd2 - Rd1
  dRflag = '**'
  if abs(dG) > 40.0:
    nG += 1
    dRflag = ''
  elif abs(dUd) > 0.01:
    nUd += 1
    dRflag = ''
  elif abs(dUq) > 0.01:
    nUq += 1
    dRflag = ''
  if dRflag == '**':
    nRd += 1

  if bPrint:
    print ('Case {:d}: dG={:.2f} dUd={:.2f} dUq={:.2f} Rd={:.2f} dRd={:.2f} {:s}'.format(idx, dG, dUd, dUq, Rd1, dRd, dRflag))

  for i in [i1, i2]:
    t = model.t[i]
    Id = y_true[i,jId] * model.normfacs['Id']['scale'] + model.normfacs['Id']['offset']
    Iq = y_true[i,jIq] * model.normfacs['Iq']['scale'] + model.normfacs['Iq']['offset']
    Vd = u[i,jVd] * model.normfacs['Vd']['scale'] + model.normfacs['Vd']['offset']
    Vq = u[i,jVq] * model.normfacs['Vq']['scale'] + model.normfacs['Vq']['offset']
    Rd = abs(Vd/Id)
    if Rd < minRd:
      minRd = Rd
    if Rd > maxRd:
      maxRd = Rd
    Rq = abs(Vq/Iq)
    if Rq < minRq:
      minRq = Rq
    if Rq > maxRq:
      maxRq = Rq
#    print ('  t={:6.3f} Id={:.3f} Iq={:6.3f} Vod={:8.3f} Voq={:8.3f} Rd={:8.2f} Rq={:8.2f}'.format (t, Id, Iq, Vod, Voq, Rd, Rq))

def clamp_loss (model, bPrint):
  """Estimates the clamping loss for a model.

  Clamping loss occurs when a forward-evaluated output channel exceeds its configured limits.
  If *clamp* is not part of the *model* configuration, there is no clamping loss. 

  Args:
    model (pv3_poly): The *pv3_poly* instance with data loaded and normalized, and the model previously exported.
    bPrint (bool): request diagnostic output of clamping loss by case and output channel.

  Returns:
    float: sum of clamping loss over all cases and output channels
  """
  if not hasattr (model, 'clamp'):
    return 0.0
  total, cases = model.clampingErrors (bByCase=True)
  total = total.detach().numpy()

  if bPrint:
    out_size = len(model.COL_Y)
    colstr = ','.join('{:s}'.format(col) for col in model.COL_Y)
    print ('Clamping Loss by Case and Output')
    print ('Idx,{:s}'.format(colstr))
    for i in range(len(cases)):
      valstr = ','.join('{:.6f}'.format(cases[i][j]) for j in range(out_size))
      print ('{:d},{:s}'.format(i, valstr))
    print ('Total Clamping Error Summary')
    for j in range(out_size):
      print ('{:4s} Loss={:8.6f}'.format (model.COL_Y[j], total[j]))
  return np.sum(total)

def build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl):
  """Assembles input values into a properly ordered array for a model's forward evaluation, excluding any not provided, as indicated by *NaN*. *Internal*

  Args:
    T0 (float): temperature, may be *NaN*
    G0 (float): solar irradiance
    F0 (float): control frequency, may be *NaN*
    Ud0 (float): direct-axis voltage control index
    Uq0 (float): quadrature-axis voltage control index
    Vd0 (float): direct-axis terminal voltage
    Vq1 (float): quadrature-axis terminal voltage
    GVrms (float): polynomial input feature
    Ctl (float): control-mode input feature

  Returns:
    list(float): array of input values for *model.steady_state_response*
  """
  if math.isnan(T0):
    if math.isnan(F0):
      return [G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
    else:
      return [G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
  elif math.isnan(F0):
    return [T0, G0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]
  return [T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl]

def sensitivity_analysis (model, bPrint, bLog = False, bAutoRange = False, dThresh=0.10):
  """Estimates the sensitivity of an exported model in z domain.

  The sensitivity is estimated for grid interface output variables (Id, Iq) with
  respect to the closed loop feedback variables (Vd, Vq). The sensitivity is examined
  over a set of operating points in *G*, *Ctl*, *Ud*, *Uq*, *Fc*, *Vd*, and *Vq*. At
  each operating point, *Vd* and *Vq* is perturbed separately by a small amount, and then
  *GVrms* is updated. The *model.steady_state_response* function is used to estimate
  the changes in *Id* and *Iq* for the perturbations in *Vd* and *Vq*. 

  Args:
    model (pv3_poly): The *pv3_poly* instance with data loaded and normalized, and the model previously exported.
    bPrint (bool): print the maximum values of the four partial derivatives of *Id*, *Iq* with respect to *Vd*, *Vq*.
    bLog (bool): print diagnostics of the operating points and perturbations over the sensitivity evaluation set.
    bAutoRange (bool): if *True*, use the minima and maxima of the *model* training set to establish the input channel bounds. If *False*, use a built-in set of bounds for the GridLink SDI lab tests.
    dThresh (float): when printing, show the number of cases within each range that have sigma exceeding this threshold.

  Yields:
    Printed output of the auto-range boundaries and the sensitivity evaluation set.

  Returns:
    float: the maximum of four partial derivatives of Id, Iq with respect to Vd, Vq.
  """
  maxdIdVd = 0.0
  maxdIdVq = 0.0
  maxdIqVd = 0.0
  maxdIqVq = 0.0
  delta = 0.01

  T_range = [math.nan]
  Fc_range = [math.nan]
  if bAutoRange:
    print ('Autoranging:')
    print ('Column       Min       Max      Mean     Range')
    idx = 0
    for c in model.COL_U + model.COL_Y:
      fac = model.normfacs[c]
      dmean = fac['offset']
      if 'max' in fac and 'min' in fac:
        dmax = fac['max']
        dmin = fac['min']
      else:
        dmax = model.de_normalize (np.max (model.data_train[:,:,idx]), fac)
        dmin = model.de_normalize (np.min (model.data_train[:,:,idx]), fac)
      drange = dmax - dmin
      if abs(drange) <= 0.0:
        drange = 1.0
      print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (c, dmin, dmax, dmean, drange))
      if c in ['G']:
        G_range = np.linspace (max (dmin, 50.0), dmax, 7) # 11)
      elif c in ['Ud', 'Md']:
        Ud_range = np.linspace (dmin, dmax, 5) # 5)
      elif c in ['Uq', 'Mq']:
        Uq_range = np.linspace (dmin, dmax, 5) # 5)
      elif c in ['Ctl', 'Ctrl', 'Step', 'Ramp']:
        Ctl_range = np.linspace (0.0, 1.0, 2)
      elif c in ['Vd']:
        Vd_range = np.linspace (dmin, dmax, 5) # 11)
      elif c in ['Vq']:
        Vq_range = np.linspace (dmin, dmax, 5) # 11)
      elif c in ['T']:
        T_range = np.linspace (dmin, dmax, 2)
      elif c in ['Fc']:
        Fc_range = np.linspace (dmin, dmax, 3) # 5)
      idx +=     1
  else:
    G_range = np.linspace (100.0, 1000.0, 10)
    Ud_range = np.linspace (0.8, 1.2, 5)
    Uq_range = np.linspace (-0.5, 0.5, 5)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Vd_range = np.linspace (170.0, 340.0, 5)
    Vq_range = np.linspace (-140.0, 140.0, 5)
  ncases = len(G_range) * len(Ud_range) * len(Uq_range) * len(Ctl_range) * len(Vd_range) * len(Vq_range) * len(T_range) * len(Fc_range)
  print ('Using {:d} sensitivity cases'.format (ncases))
  print (' T_range', T_range)
  print (' Fc_range', Fc_range)
  print (' G_range', G_range)
  print (' Ud_range', Ud_range)
  print (' Uq_range', Uq_range)
  print (' Ctl_range', Ctl_range)
  print (' Vd_range', Vd_range)
  print (' Vq_range', Vq_range)
  G_count = np.zeros(G_range.shape[0])
  T_count = np.zeros(T_range.shape[0])
  Fc_count = np.zeros(Fc_range.shape[0])
  Ctl_count = np.zeros(Ctl_range.shape[0])
  Ud_count = np.zeros(Ud_range.shape[0])
  Uq_count = np.zeros(Uq_range.shape[0])
  Vd_count = np.zeros(Vd_range.shape[0])
  Vq_count = np.zeros(Vq_range.shape[0])

  if bLog:
    print ('   G0   Ud0   Uq0   Vd0    Vq0 Ctl   dIdVd   dIdVq   dIqVd   dIqVq')
  model.start_simulation (bPrint=False)
  for T0 in T_range:
    T_idx = np.nonzero(T_range==T0)[0][0]
    for F0 in Fc_range:
      Fc_idx = np.nonzero(Fc_range==F0)[0][0]
      for G0 in G_range:
        G_idx = np.nonzero(G_range==G0)[0][0]
        print ('checking G={:.2f}'.format(G0), 'T=', T0, 'Fc=', F0)
        for Ud0 in Ud_range:
          Ud_idx = np.nonzero(Ud_range==Ud0)[0][0]
          for Uq0 in Uq_range:
            Uq_idx = np.nonzero(Uq_range==Uq0)[0][0]
            for Ctl in Ctl_range:
              Ctl_idx = np.nonzero(Ctl_range==Ctl)[0][0]
              for Vd0 in Vd_range:
                Vd_idx = np.nonzero(Vd_range==Vd0)[0][0]
                for Vq0 in Vq_range:
                  Vq_idx = np.nonzero(Vq_range==Vq0)[0][0]
                  # baseline
                  Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq0*Vq0)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq0, GVrms, Ctl)
                  Vdc0, Idc0, Id0, Iq0 = model.steady_state_response (step_vals)

                  # change Vd and GVrms
                  Vd1 = Vd0 + delta
                  Vrms = KRMS * math.sqrt(Vd1*Vd1 + Vq0*Vq0)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd1, Vq0, GVrms, Ctl)
                  Vdc1, Idc1, Id1, Iq1 = model.steady_state_response (step_vals)

                  # change Vq and GVrms
                  Vq1 = Vq0 + delta
                  Vrms = KRMS * math.sqrt(Vd0*Vd0 + Vq1*Vq1)
                  GVrms = G0 * Vrms
                  step_vals = build_step_vals (T0, G0, F0, Ud0, Uq0, Vd0, Vq1, GVrms, Ctl)
                  Vdc2, Idc2, Id2, Iq2 = model.steady_state_response (step_vals)

                  # calculate the changes
                  dIdVd = (Id1 - Id0) / delta
                  dIqVd = (Iq1 - Iq0) / delta
                  dIdVq = (Id2 - Id0) / delta
                  dIqVq = (Iq2 - Iq0) / delta
                  if bLog:
                    print ('{:6.1f} {:4.2f} {:5.2f} {:5.1f} {:6.1f} {:3.1f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format (G0, Ud0, Uq0, Vd0, Vq0, Ctl,
                                                                                                                     dIdVd, dIdVq, dIqVd, dIqVq))
                  # track the global maxima
                  dIdVd = abs(dIdVd)
                  if dIdVd > maxdIdVd:
                    maxdIdVd = dIdVd

                  dIdVq = abs(dIdVq)
                  if dIdVq > maxdIdVq:
                    maxdIdVq = dIdVq

                  dIqVd = abs(dIqVd)
                  if dIqVd > maxdIqVd:
                    maxdIqVd = dIqVd

                  dIqVq = abs(dIqVq)
                  if dIqVq > maxdIqVq:
                    maxdIqVq = dIqVq

                  # am I above dThresh
                  if dIdVd >= dThresh or dIdVq >= dThresh or dIqVd >= dThresh or dIqVq >= dThresh:
                    T_count[T_idx] += 1.0
                    G_count[G_idx] += 1.0
                    Ctl_count[Ctl_idx] += 1.0
                    Fc_count[Fc_idx] += 1.0
                    Ud_count[Ud_idx] += 1.0
                    Uq_count[Uq_idx] += 1.0
                    Vd_count[Vd_idx] += 1.0
                    Vq_count[Vq_idx] += 1.0
  if bPrint:
    print ('                                     dIdVd   dIdVq   dIqVd   dIqVq')
    print ('{:34s} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format ('Maximum Magnitudes', maxdIdVd, maxdIdVq, maxdIqVd, maxdIqVq))
    print ('Counting instances of max sensitivity >= {:7.4f}'.format (dThresh))
    print ('  T counts:  ', T_count)
    print ('  G counts:  ', G_count)
    print ('  Ctl counts:', Ctl_count)
    print ('  Fc counts: ', Fc_count)
    print ('  Ud counts: ', Ud_count)
    print ('  Uq counts: ', Uq_count)
    print ('  Vd counts: ', Vd_count)
    print ('  Vq counts: ', Vq_count)
  return max(maxdIdVd, maxdIdVq, maxdIqVd, maxdIqVq)

def find_channel_indices (targets, available):
  """Pick out the training dataset channel numbers used in sensitivity evaluations. Call separately for the input channels and the output channesl. *Internal*

  Args:
    targets (list(str)): array of channel names used in the sensitivity evaluation set
    available (list(str)): array of channel names available in a model's training dataset

  Returns:
    int: length of the next return array, equal to *len(targets)*
    list(int): array of training dataset channel numbers
  """
  n = len(targets)
  idx = []
  for i in range(n):
    idx.append (available.index(targets[i]))
  return n, idx

counter = 0
"""Tracks the number of recursive calls to *build_baselines*
"""

def build_baselines (bases, step_vals, cfg, keys, indices, lens, level):
  """Recursive function to add a set of operating points to the sensitivity evaluation set. Uses a depth-first approach. When the last channel number is processed, the recursion will back up to a previous channel number that was not fully processed yet. *Internal*

  Args:
    bases (list(float)[]): array of *step_vals* for operating points in the sensitivity evaluation set 
    step_vals (list(float)): input channel values for a *pv3_poly* model steady-state operating point 
    cfg (dict): the *sensitivity* member of a *pv3_poly* configuration, which incluces a member *sets* contained keyed channel names
    keys (list(str)): list of channel names from the *sets* member of *cfg*, each of these corresponds to a *level* of recursion
    indices (list(int)): keeps track of the channel number to resume processing whenever *level* reachs the last *key* 
    lens (list(int)): the number of operating point values for each named channel in *keys*
    level (int): enters with 0, backs up at the length of *keys* minus 1

  Yields:
    Appending to *bases*. Updates *counter* in each call.

  Returns:
    None
  """
  global counter
  counter += 1

  key = keys[level]
  ary = cfg['sets'][key]
  idx = cfg['idx_set'][key]

  if level+1 == len(keys): # add basecases at the lowest level
    for i in range(lens[level]):
      step_vals[idx] = ary[i]
      bases.append (step_vals.copy())
  else: # propagate this new value down to lower levels
    step_vals[idx] = ary[indices[level]]

  if level+1 < len(keys):
    level += 1
    build_baselines (bases, step_vals, cfg, keys, indices, lens, level)
  else:
    level -= 1
    while level >= 0:
      if indices[level]+1 >= lens[level]:
        level -= 1
      else:
        indices[level] += 1
        indices[level+1:] = 0
        build_baselines (bases, step_vals, cfg, keys, indices, lens, level)

def get_gvrms (G, Vd, Vq, k):
  """Calculates the polynomial feature *GVrms*. *Internal*

  Args:
    G (float): solar irradiance.
    Vd (float): direct-axis voltage.
    Vq (float): quadrature-axis voltage.
    k (float): sqrt(1.5) for three-phase inverters, 1.0 for single-phase inverters.

  Returns:
    float: value of *GVrms*
  """
  return G * k * math.sqrt(Vd*Vd + Vq*Vq)

def model_sensitivity (model, bPrint):
  """Calculates the maximum sensitivity of an exported *Norton* model in z domain.

  The model configuration must include a *sensitivy* structure that was used
  in training. This determines the sensitivity evaluation set.

  See Also:
    :func:`sensitivity_analysis`

  Args:
    model (pv3_poly): The *pv3_poly* instance with data loaded and normalized, and the model previously exported.
    bPrint (bool): print the maximum sensitivity (return value).

  Yields:
    Printed information about the sensitivity evaluation set and columns.

  Returns:
    float: The maximum derivative of *Id*, *Iq* with respect to *Vd*, *Vq*.
  """
  cfg = model.sensitivity
  cfg['n_in'], cfg['idx_in'] = find_channel_indices (cfg['inputs'], model.COL_U)
  cfg['n_out'], cfg['idx_out'] = find_channel_indices (cfg['outputs'], model.COL_Y)
  cfg['idx_set'] = {}
  for key in cfg['sets']:
    cfg['idx_set'][key] = model.COL_U.index(key)
  print ('inputs', cfg['inputs'], cfg['idx_in'])
  print ('outputs', cfg['outputs'], cfg['idx_out'])
  print ('sets', cfg['idx_set'])
  cfg['idx_g_rms'] = model.COL_U.index (cfg['GVrms']['G'])
  cfg['idx_vd_rms'] = model.COL_U.index (cfg['GVrms']['Vd'])
  cfg['idx_vq_rms'] = model.COL_U.index (cfg['GVrms']['Vq'])
  cfg['idx_gvrms'] = model.COL_U.index ('GVrms')
  krms = cfg['GVrms']['k']
  print ('GVrms', cfg['idx_g_rms'], cfg['idx_vd_rms'], cfg['idx_vq_rms'], krms, cfg['idx_gvrms'])

  max_sens = np.zeros ((cfg['n_out'], cfg['n_in']))
  sens = np.zeros ((cfg['n_out'], cfg['n_in']))
  step_vals = np.zeros (len(model.COL_U))

  bases = []
  keys = list(cfg['sets'])
  indices = np.zeros(len(keys), dtype=int)
  lens = np.zeros(len(keys), dtype=int)
  for i in range(len(keys)):
    lens[i] = len(cfg['sets'][keys[i]])
  build_baselines (bases, step_vals, cfg, keys, indices, lens, 0)
  print (len(bases), 'base cases', counter, 'function calls')

  model.start_simulation (bPrint=False)
  for vals in bases:
    Vd0 = vals[cfg['idx_vd_rms']]
    Vq0 = vals[cfg['idx_vq_rms']]
    Vd1 = Vd0 + 1.0
    Vq1 = Vq0 + 1.0

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd0, Vq0, krms)
    _, _, Id0, Iq0 = model.steady_state_response (vals.copy())
    #print (vals, Id0, Iq0)

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd1, Vq0, krms)
    vals[cfg['idx_vd_rms']] = Vd1
    vals[cfg['idx_vq_rms']] = Vq0
    _, _, Id1, Iq1 = model.steady_state_response (vals.copy())
    #print (vals, Id1, Iq1)

    vals[cfg['idx_gvrms']] = get_gvrms (vals[cfg['idx_g_rms']], Vd0, Vq1, krms)
    vals[cfg['idx_vd_rms']] = Vd0
    vals[cfg['idx_vq_rms']] = Vq1
    _, _, Id2, Iq2 = model.steady_state_response (vals.copy())
    #print (vals, Id2, Iq2)
    #prevent aliasing the base cases
    vals[cfg['idx_vq_rms']] = Vq0

    sens[0][0] = abs(Id1 - Id0)
    sens[1][0] = abs(Iq1 - Iq0)
    sens[0][1] = abs(Id2 - Id0)
    sens[1][1] = abs(Iq2 - Iq0)
    for i in range(2):
      for j in range(2):
        if sens[i][j] > max_sens[i][j]:
          max_sens[i][j] = sens[i][j]

  if bPrint:
    print (max_sens)
  return np.max(max_sens)

def thevenin_sensitivity_analysis (model, bPrint, bLog = False, bReducedSet = False):
  """Calculates the maximum sensitivity of an exported *Thevenin* model in z domain.

  This function uses a fixed sensitivity evaluation set for the GridLink SDI lab tests.

  See Also:
    :func:`sensitivity_analysis`

  Args:
    model (pv3_poly): The *pv3_poly* instance with data loaded and normalized, and the model previously exported.
    bPrint (bool): print the maximum values of each partial derivative of *Vd*, *Vq* with respect to *Id*, *Iq*
    bLog (bool): print the four partial derivatives of *Vd*, *Vq*, w.r.t. *Id*, *Iq* sat each operating point in the sensitivity evaluation set.
    bReducedSet (bool): use a reduced sensitivity evaluation set of 72 cases instead of the full set of 60,500 cases

  Yields:
    Printed information about the sensitivity evaluation set and columns.

  Returns:
    float: The maximum derivative of *Vd*, *Vq* with respect to *Id*, *Iq*.
  """
  maxdVdId = 0.0
  maxdVdIq = 0.0
  maxdVqId = 0.0
  maxdVqIq = 0.0
  delta = 0.01

  if bReducedSet:
    # 72 cases for faster training
    G_range = np.linspace (300.0, 1000.0, 3)
    Ud_range = np.linspace (1.0, 1.1, 2)
    Uq_range = np.linspace (-0.5, 0.5, 2)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Id_range = np.linspace (3.0, 6.0, 3)
    Iq_range = np.linspace (-2.7, -2.7, 1)
  else:
    # 60500 cases for more accuracy
    G_range = np.linspace (100.0, 1000.0, 10)
    Ud_range = np.linspace (0.8, 1.2, 5)
    Uq_range = np.linspace (-0.5, 0.5, 5)
    Ctl_range = np.linspace (0.0, 1.0, 2)
    Id_range = np.linspace (-0.1, 6.0, 11)
    Iq_range = np.linspace (-2.7, 2.4, 11)

  ncases = len(G_range) * len(Ud_range) * len(Uq_range) * len(Ctl_range) * len(Id_range) * len(Iq_range)
  bGVrms = 'GVrms' in model.COL_U
  bGIrms = 'GIrms' in model.COL_U
  worst_dVdId = None
  worst_dVdIq = None
  worst_dVqId = None
  worst_dVqIq = None
  print ('Using {:d} sensitivity cases with dI={:.3f}, GVrms {:s}, GIrms {:s}'.format (ncases, delta, str (bGVrms), str (bGIrms)))
  if bLog:
    print ('   G0   Ud0   Uq0   Id0    Iq0 Ctl   dVdId   dVdIq   dVqId   dVqIq')
  model.start_simulation (bPrint=False)
  for G0 in G_range:
    print (' checking G0={:.2f}'.format (G0))
    for Ud0 in Ud_range:
      for Uq0 in Uq_range:
        # estimate GVrms, ignoring any feedback effects from Id and Iq
        Vdx = np.interp (Ud0, [0.8, 1.21], [170.0, 350.0])
        Vqx = np.interp (Uq0, [-0.5, 0.5], [-154.0, 144.0])
        Vrms = KRMS * math.sqrt(Vdx*Vdx + Vqx*Vqx)
        GVrms = G0 * Vrms
        for Ctl in Ctl_range:
          for Id0 in Id_range:
            for Iq0 in Iq_range:
              # baseline
              Irms = KRMS * math.sqrt(Id0*Id0 + Iq0*Iq0)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id0, Iq0, Ctl]
              Vdc0, Idc0, Vd0, Vq0 = model.steady_state_response (step_vals)

              # change Id
              Id1 = Id0 + delta
              Irms = KRMS * math.sqrt(Id1*Id1 + Iq0*Iq0)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id1, Iq0, Ctl]
              Vdc1, Idc1, Vd1, Vq1 = model.steady_state_response (step_vals)

              # change Iq
              Iq1 = Iq0 + delta
              Irms = KRMS * math.sqrt(Id0*Id0 + Iq1*Iq1)
              GIrms = G0 * Irms
              if bGIrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, GIrms, Ctl]
              elif bGVrms:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, GVrms, Ctl]
              else:
                step_vals = [G0, Ud0, Uq0, Id0, Iq1, Ctl]
              Vdc2, Idc2, Vd2, Vq2 = model.steady_state_response (step_vals)

              # calculate the changes
              dVdId = (Vd1 - Vd0) / delta
              dVqId = (Vq1 - Vq0) / delta
              dVdIq = (Vd2 - Vd0) / delta
              dVqIq = (Vq2 - Vq0) / delta
              if bLog:
                print ('{:6.1f} {:4.2f} {:5.2f} {:5.1f} {:6.1f} {:3.1f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format (G0, Ud0, Uq0, Id0, Iq0, Ctl,
                                                                                                                 dVdId, dVdIq, dVqId, dVqIq))
              # track the global maxima
              dVdId = abs(dVdId)
              if dVdId > maxdVdId:
                maxdVdId = dVdId
                worst_dVdId = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVdIq = abs(dVdIq)
              if dVdIq > maxdVdIq:
                maxdVdIq = dVdIq
                worst_dVdIq = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVqId = abs(dVqId)
              if dVqId > maxdVqId:
                maxdVqId = dVqId
                worst_dVqId = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

              dVqIq = abs(dVqIq)
              if dVqIq > maxdVqIq:
                maxdVqIq = dVqIq
                worst_dVqIq = [G0, Ud0, Uq0, Id0, Iq0, Ctl]

  if bPrint:
    print ('                                     dVdId   dVdIq   dVqId   dVqIq')
    print ('{:34s} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format ('Maximum Magnitudes', maxdVdId, maxdVdIq, maxdVqId, maxdVqIq))
    print ('worst case dVdId [G, Ud, Uq, Id, Iq]:', worst_dVdId)
    print ('worst case dVdIq [G, Ud, Uq, Id, Iq]:', worst_dVdIq)
    print ('worst case dVqId [G, Ud, Uq, Id, Iq]:', worst_dVqId)
    print ('worst case dVqIq [G, Ud, Uq, Id, Iq]:', worst_dVqIq)
  return max(maxdVdId, maxdVdIq, maxdVqId, maxdVqIq)

