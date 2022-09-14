# copyright 2021-2022 Battelle Memorial Institute
# exports a trained HW model and normalization coefficients to a single JSON file
#  arg1: relative path to trained model configuration file
#  arg2: relative path to the combined output file

import os
import sys
import pv3_poly as pv3_model

model_path = './big/balanced_config.json'
model_path = './flatstable/flatstable_config.json'

if __name__ == '__main__':
  if len(sys.argv) > 1:
    model_path = sys.argv[1]
  model_folder, config_file = os.path.split(model_path)
  model_root = config_file.rstrip('.json')
  model_root = model_root.rstrip('_config')
  export_path = os.path.join(model_folder,'{:s}_fhf.json'.format(model_root))
  if len(sys.argv) > 2:
    export_path = sys.argv[2]

  print ('Read Model from:', model_path)
  print ('Export Model to:', export_path)

  model = pv3_model.pv3(training_config=model_path)
  model.loadNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  model.exportModel(export_path)
  model.check_poles()
#  model.printStateDicts()

