import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rcParams['savefig.directory'] = os.getcwd()

data_path = 'ucfB_t_2nd_Ctrl'

if __name__ == '__main__':
  if len(sys.argv) > 1:
    data_path = sys.argv[1]

  fname = os.path.join (data_path, 'Loss.npy')
  data = np.load (fname)

  print ('Read {:d} epochs from {:s}'.format (len(data[0]), fname))
  print ('Data shape is', data.shape)
  print ('Last training loss {:.6f}'.format (data[0][-1]))
  print ('Last validation loss {:.6f}'.format (data[1][-1]))

  plt.figure()
  plt.title(fname)
  plt.plot(np.log10(data[0]), label='Training Loss')
  plt.plot(np.log10(data[1]), label='Validation Loss')
  if data.shape[0] > 2:
    plt.plot(np.log10(data[2]), label='Sensitivity Loss')
  plt.ylabel ('Log10')
  plt.xlabel ('Epoch')
  plt.legend()
  plt.grid(True)
  plt.show()

