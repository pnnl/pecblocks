# copyright 2021-2024 Battelle Memorial Institute
# supervises training of HW model for a three-phase inverter
#  arg1: relative path to the model configuration file

import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import pecblocks.pv3_poly as pv3_model

bWantMAE = False
bWantPlot = False

if __name__ == '__main__':
  if len(sys.argv) > 1:
    config_file = sys.argv[1]
    fp = open (config_file, 'r')
    cfg = json.load (fp)
    fp.close()
    data_path = cfg['data_path']
    model_folder = cfg['model_folder']
    model_root = cfg['model_root']
  else:
    print ('Usage: python pv3_training.py config.json')
    quit()

  print ('model_folder =', model_folder)
  print ('model_root =', model_root)
  print ('data_path =', data_path)

  model = pv3_model.pv3(training_config=config_file)
  model.loadTrainingData(data_path)
  model.applyAndSaveNormalization()
  model.initializeModelStructure()
  train_time, LOSS, VALID, SENS = model.trainModelCoefficients(bMAE=False)
  model.saveModelCoefficients()
  rmse, mae, case_rmse, case_mae = model.trainingErrors(bByCase=False)

  nlookback = 10
  recent_loss = LOSS[len(LOSS)-nlookback:]

  fp = open (os.path.join (model_folder, 'summary.txt'), 'w')
  jp = open (os.path.join (model_folder, 'normfacs.json'), 'r')
  normfacs = json.load (jp)
  jp.close()
  print ('Dataset Summary: (Mean=Offset, Range=Scale)', file=fp)
  print ('Column       Min       Max      Mean     Range', file=fp)
  for key, row in normfacs.items():
    print ('{:6s} {:9.3f} {:9.3f} {:9.3f} {:9.3f}'.format (key, row['min'], row['max'], row['offset'], row['scale']), file=fp)

  out_size = len(model.COL_Y)
  recent_loss = np.mean(recent_loss)
  valstr = ' '.join('{:.4f}'.format(rmse[j]) for j in range(out_size))
  print ('COL_Y', model.COL_Y, file=fp)
  print ('Train time: {:.2f}, Recent loss: {:.6f}, RMS Errors: {:s}'.format (train_time, recent_loss, valstr), file=fp)
  print ('COL_Y', model.COL_Y)
  print ('Train time: {:.2f}, Recent loss: {:.6f}, RMS Errors: {:s}'.format (train_time, recent_loss, valstr))
  if bWantMAE:
    valstr = ' '.join('{:.4f}'.format(mae[j]) for j in range(out_size))
    print ('                          MAE Errors: {:s}'.format (valstr), file=fp)
  fp.close()

  if bWantPlot:
    plt.figure()
    plt.title(model_root)
    plt.plot(np.log10(LOSS), label='Training Loss')
    plt.plot(np.log10(VALID), label='Validation Loss')
    if np.min(SENS) > 0.0:
      plt.plot(np.log10(SENS), label='Sensitivity Loss')
    plt.ylabel ('Log10')
    plt.xlabel ('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_folder, '{:s}_train_loss.pdf'.format(model_root)))
    plt.show()

