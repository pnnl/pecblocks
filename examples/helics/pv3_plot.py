# copyright 2021-2024 Battelle Memorial Institute
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['savefig.directory'] = os.getcwd()

df = pd.read_hdf ('pv3_server.hdf5')
#print (df)
#df['Vs'] = np.abs(df['Vs'])
#df['Vc'] = np.abs(df['Vc'])
#df['Is'] = np.abs(df['Is'])
#df['Ic'] = np.abs(df['Ic'])
#print (df)

#print (df.columns.values)
print (df.head(10))
print (df.tail(1))
#df.plot(x='t', y=['Vdc', 'Idc'], backend='matplotlib', subplots=True)
df.plot(x='t', y=['G', 'T', 'Md', 'Mq', 'Fc', 'Ctl', 'Vd', 'Vq', 'GVrms', 'Vdc', 'Idc', 'Id', 'Iq'], 
        layout=(3, 5), figsize=(18,10), subplots=True)

plt.show()
