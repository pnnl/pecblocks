import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['savefig.directory'] = os.getcwd()

data = np.load ('lab1/loss.npy')

plt.figure()
plt.plot(data[0], label='Training Loss')
plt.plot(data[1], label='Validation Loss')
#plt.set_xlabel ('Epoch')
plt.legend()
plt.grid(True)
plt.show()

