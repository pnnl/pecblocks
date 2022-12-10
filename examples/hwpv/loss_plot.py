import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['savefig.directory'] = os.getcwd()

data = np.load ('lab1/loss.npy')

plt.figure()
plt.plot(data)
plt.grid(True)
plt.show()

