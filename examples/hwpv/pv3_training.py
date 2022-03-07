import numpy as np
import os
import matplotlib.pyplot as plt
import pv3_poly as pv3_model

data_path = r'./data/gfm8.hdf5'
model_folder = r'./models'

if __name__ == '__main__':

  model = pv3_model.pv3(os.path.join(model_folder,'gfm8_config.json'))
  model.loadTrainingData(data_path)
  model.applyAndSaveNormalization(model_folder)
  model.initializeModelStructure()
  train_time, LOSS = model.trainModelCoefficients()
  model.saveModelCoefficients(model_folder)
#  quit()
  rmse, mae, case_rmse = model.trainingErrors(False)

  nlookback = 10 * int(model.n_cases / model.batch_size)
  recent_loss = LOSS[len(LOSS)-nlookback:]
#  print (nlookback, recent_loss)
  print ('COL_Y', model.COL_Y)
  valstr = ' '.join('{:.4f}'.format(rmse[col]) for col in model.COL_Y)
  print ('Train time: {:.2f}, Recent loss: {:.2f}, RMS Errors: {:s}'.format (train_time, 
    np.mean(recent_loss), valstr))

  plt.figure()
  plt.plot(LOSS)
  plt.grid(True)
  plt.savefig(os.path.join(model_folder, 'GFM8_train_loss.pdf'))
  plt.show()