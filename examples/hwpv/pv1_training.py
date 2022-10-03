import numpy as np
import os
import matplotlib.pyplot as plt
import pv1_poly as pv1_model
#import pv1_fhf as pv1_model
#import pv1_model
#import pv1_feedback as pv1_model
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

if __name__ == '__main__':

  model = pv1_model.pv1(os.path.join(model_folder,'pv1_config.json'))
  model.loadTrainingData(data_path)
  model.applyAndSaveNormalization(model_folder)
  model.initializeModelStructure()
  train_time, LOSS = model.trainModelCoefficients(bMAE=False)
  model.saveModelCoefficients(model_folder)
#  quit()
  rmse, mae, case_rmse, case_mae = model.trainingErrors(False)

  nlookback = 10 * int(model.n_cases / model.batch_size)
  recent_loss = LOSS[len(LOSS)-nlookback:]
#  print (nlookback, recent_loss)
#  print ('COL_Y', model.COL_Y)
  valstr = ' '.join('{:.4f}'.format(rmse[col]) for col in model.COL_Y)
  print ('Train time: {:.2f}, Recent loss: {:.2f}, RMS Errors: {:s}'.format (train_time, 
    np.mean(recent_loss), valstr))
  valstr = ' '.join('{:.4f}'.format(mae[col]) for col in model.COL_Y)
  print ('                          MAE Errors: {:s}'.format (valstr))

  plt.figure()
  plt.plot(LOSS)
  plt.grid(True)
  plt.savefig(os.path.join(model_folder, 'FHF_train_loss.pdf'))
  plt.show()