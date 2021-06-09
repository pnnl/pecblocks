import os
import sys
#import pecblocks.util
import json
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

model_sets = [
   {'name':'test','blocks':'F_nl+G2'},
   {'name':'GtoVdc','blocks':'F1+G1+F2'},
   {'name':'GtoIdc','blocks':'F1+G1+F2'},
   {'name':'TtoVdc','blocks':'F1+G1+F2'},
   {'name':'TtoIdc','blocks':'F1+G1+F2'},
   {'name':'VdctoIdc','blocks':'F1+G1+F2'},
   {'name':'VdctoPac','blocks':'F1+G1+F2'},
   {'name':'VdctoQac','blocks':'F1+G1+F2'}
]

if __name__ == '__main__':
    json_file = 'models.json'
    model_name = 'GtoVdc'
    with open (json_file, 'r') as read_file:
        models = json.load (read_file)
    if model_name not in models:
        print ('Model {:s} was not found in {:s}'.format (model_name, json_file))
        quit()

    mdl = models[model_name]
    block = 'G1'
    nk = mdl[block]['n_k']
    na = len(mdl[block]['denominator']) + 1  # a0 always 1
    nb = len(mdl[block]['numerator']) + nk  # first nk coefficients are zero

    a = np.ones(na)
    b = np.zeros(nb)
    a[1:] = mdl[block]['denominator']
    b[nk:] = mdl[block]['numerator']

    print ('A (den)', a)
    print ('B (num)', b)

#    butter = signal.dlti(*signal.butter(3, 0.5))
#    t, y = signal.dimpulse(butter, n=25)
    t, y = signal.dimpulse([b, a, 0.02], n=25)
    plt.step(t, np.squeeze(y))
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    plt.show()
