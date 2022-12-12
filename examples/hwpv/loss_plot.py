import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rcParams['savefig.directory'] = os.getcwd()

data_path = 'lab1'

if __name__ == '__main__':
  if len(sys.argv) > 1:
    data_path = sys.argv[1]

  fname = os.path.join (data_path, 'loss.npy')
  data = np.load (fname)

  plt.figure()
  plt.title(fname)
  plt.plot(np.log10(data[0]), label='Training Loss')
  plt.plot(np.log10(data[1]), label='Validation Loss')
  plt.ylabel ('Log10')
  plt.xlabel ('Epoch')
  plt.legend()
  plt.grid(True)
  plt.show()

