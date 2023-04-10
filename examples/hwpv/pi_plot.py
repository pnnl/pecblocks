import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['savefig.directory'] = os.getcwd()

df = pd.read_hdf ('hwpv_pi.hdf5')
print (df.head(10))
print (df.tail(1))
df.plot(x='t', y=['G', 'T', 'Md', 'Mq', 'Fc', 'Ctl', 'Vrms', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq'], 
        layout=(3, 4), figsize=(16,10), subplots=True)

plt.show()
