import os
import sys
import pv1_poly as pv1_model

model_folder = r'./'

if __name__ == '__main__':
  model = pv1_model.pv1 ()
  model.load_sim_config (os.path.join(model_folder,'pv1_fhf_poly.json'))

