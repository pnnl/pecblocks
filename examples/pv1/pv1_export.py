import os
import sys
import pv1_poly as pv1_model

data_path = r'./data/pv1.hdf5'
model_folder = r'./models'

if __name__ == '__main__':
  model = pv1_model.pv1(os.path.join(model_folder,'pv1_config.json'))
#  model.loadTrainingData(data_path)
  model.loadNormalization(os.path.join(model_folder,'normfacs.json'))
  model.initializeModelStructure()
  model.loadModelCoefficients(model_folder)
  model.exportModel(os.path.join(model_folder,'pv1_fhf_poly.json'))

