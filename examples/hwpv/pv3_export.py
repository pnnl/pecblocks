import os
import sys
import pv3_poly as pv3_model

root = 'tacs' # 'unbalanced' # 'gfm8'
model_folder = r'./models'

if __name__ == '__main__':
  model = pv3_model.pv3(os.path.join(model_folder,'{:s}_config.json'.format(root)))
  model.loadNormalization(os.path.join(model_folder,'normfacs.json'))
  model.initializeModelStructure()
  model.loadModelCoefficients(model_folder)
  model.exportModel(os.path.join(model_folder,'{:s}_fhf.json'.format(root)))

