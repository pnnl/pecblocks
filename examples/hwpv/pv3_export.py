# copyright 2021-2024 Battelle Memorial Institute
# exports a trained HW model and normalization coefficients to a single JSON file
#  arg1: relative path to the model configuration file

import os
import sys
import json
import pecblocks.pv3_poly as pv3_model

if __name__ == '__main__':
  if len(sys.argv) > 1:
    config_file = sys.argv[1]
    fp = open (config_file, 'r')
    cfg = json.load (fp)
    fp.close()
    model_folder = cfg['model_folder']
    model_root = cfg['model_root']
    data_path = cfg['data_path']
  else:
    print ('Usage: python pv3_export.py config.json')
    quit()
# model_root = config_file.rstrip('.json')
# print (config_file, model_root)
# model_root = model_root.rstrip('_config')
# print (model_root)
  export_path = os.path.join(model_folder,'{:s}_fhf.json'.format(model_root))
  if len(sys.argv) > 2:
    export_path = sys.argv[2]

  print ('Read Model from:', model_folder)
  print ('Export Model to:', export_path)

  model = pv3_model.pv3(training_config=config_file)
  model.loadNormalization()
  model.initializeModelStructure()
  model.loadModelCoefficients()
  model.exportModel(export_path)
  model.check_poles()
#  model.printStateDicts()

