# copyright 2021-2022 Battelle Memorial Institute
# supervises training of HW model for a three-phase inverter
#  arg1: relative path to trained model configuration file
#  arg2: relative path to the training data file, HDF5

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

#data_path = './data/flatbalanced.hdf5'
#model_path = './flatbal_continuation/flatbal_continuation_config.json'

#data_path = './data/balanced_vdvq.hdf5'
#model_path = './dc/dc_config.json'
#model_path = './flat_vdvq/flat_vdvq_config.json'

#model_path = './flatbal/flatbal_config.json'
#model_path = './flatstable/flatstable_config.json'

#data_path = '../simscape/balanced.hdf5'
#model_path = '../simscape/balanced_config.json'

#data_path = '../../../atptools/unbalanced.hdf5'
#model_path = './tacs/tacs_config.json'
#model_path = './unbal/unbal_config.json'

#data_path = './data/osg_vrms.hdf5'
#model_path = './osg_vrms/osg_vrms_config.json'

#data_path = 'c:/data/osg4_vdvq.hdf5'
#model_path = './osg4_vdvq/osg4_vdvq_config.json'

data_path = 'c:/data/ucf2.hdf5'
model_path = './ucf2/ucf2_config.json'
model_path = './ucf2ac/ucf2ac_config.json'

if __name__ == '__main__':
  if len(sys.argv) > 1:
    model_path = sys.argv[1]
    if len(sys.argv) > 2:
      data_path = sys.argv[2]

  model_folder, config_file = os.path.split(model_path)
  model_root = config_file.rstrip('.json')
  model_root = model_root.rstrip('_config')
  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)

  model = pv3_model.pv3(training_config=model_path)
  model.loadTrainingData(data_path)
  model.applyAndSaveNormalization()
  model.initializeModelStructure()
  train_time, LOSS = model.trainModelCoefficients(bMAE=False)
  model.saveModelCoefficients()
#  quit()
  rmse, mae, case_rmse, case_mae = model.trainingErrors(bByCase=False)

  nlookback = 10 #  * int(model.n_cases / model.batch_size)
  recent_loss = LOSS[len(LOSS)-nlookback:]
#  print (nlookback, recent_loss)
  print ('COL_Y', model.COL_Y)
  valstr = ' '.join('{:.4f}'.format(rmse[col]) for col in model.COL_Y)
  print ('Train time: {:.2f}, Recent loss: {:.6f}, RMS Errors: {:s}'.format (train_time, 
    np.mean(recent_loss), valstr))
  valstr = ' '.join('{:.4f}'.format(mae[col]) for col in model.COL_Y)
  print ('                          MAE Errors: {:s}'.format (valstr))

  plt.figure()
  plt.plot(LOSS)
  plt.grid(True)
  plt.savefig(os.path.join(model_folder, '{:s}_train_loss.pdf'.format(model_root)))
  plt.show()