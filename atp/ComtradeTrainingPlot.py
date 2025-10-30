# Copyright (C) 2018-2021 Battelle Memorial Institute
# file: ComtradeTrainingPlot.py
""" Plots the ATP simulation results from COMTRADE files.

Paragraph.

Public Functions:
    :main: does the work
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import h5utils

#print (plt.gcf().canvas.get_supported_filetypes())
#quit()
def make_plot(chan, case_title, file_name=None):
  t = chan['t']
  fig, ax = plt.subplots(2, 4, sharex = 'col', figsize=(16,6), constrained_layout=True)
  fig.suptitle ('Case: ' + case_title)

  ax[0,0].set_title ('Weather')
  ax[0,0].set_ylabel ('W/m2')
  lns1 = ax[0,0].plot(t, chan['G'], label='G', color='r')
  ax2 = ax[0,0].twinx()
  ax2.set_ylabel('C')
  lns2 = ax2.plot(t, chan['T'], label='T', color='b')
  lns = lns1+lns2
  labs = [l.get_label() for l in lns]
  ax[0,0].grid()
  ax[0,0].legend(lns, labs, loc='best')

  ax[0,1].set_title ('DC Bus')
  ax[0,1].set_ylabel ('V')
  lns1 = ax[0,1].plot(t, chan['Vdc'], label='Vdc', color='r')
  ax2 = ax[0,1].twinx()
  ax2.set_ylabel('A')
  lns2 = ax2.plot(t, chan['Idc'], label='Idc', color='b')
  lns = lns1+lns2
  labs = [l.get_label() for l in lns]
  ax[0,1].grid()
  ax[0,1].legend(lns, labs, loc='best')

  ax[0,2].set_title ('RMS Voltage Mag')
  ax[0,2].set_ylabel ('V')
  ax[0,2].plot(t, chan['VmagA'], color='k')
  ax[0,2].plot(t, chan['VmagB'], color='r')
  ax[0,2].plot(t, chan['VmagC'], color='b')
  ax[0,2].grid()

  ax[0,3].set_title ('Voltage Angle')
  ax[0,3].set_ylabel ('rad')
  ax[0,3].plot(t, chan['VangA'], color='k')
  ax[0,3].plot(t, chan['VangB'], color='r')
  ax[0,3].plot(t, chan['VangC'], color='b')
  ax[0,3].grid()

  ax[1,0].set_title ('Power')
  ax[1,0].set_ylabel ('kva')
  ax[1,0].plot(t, 0.001 * chan['P'], label='P', color='r')
  ax[1,0].plot(t, 0.001 * chan['Q'], label='Q', color='b')
  ax[1,0].grid()
  ax[1,0].legend()

  ax[1,1].set_title ('Frequency')
  ax[1,1].set_ylabel ('Hz')
  ax[1,1].plot(t, chan['F0'], label='Nominal', color='k')
  ax[1,1].plot(t, chan['Fc'], label='Control', color='r')
  ax[1,1].plot(t, chan['F'], label='Measured', color='b')
  ax[1,1].grid()
  ax[1,1].legend()

  ax[1,2].set_title ('RMS Current Mag')
  ax[1,2].set_ylabel ('A')
  ax[1,2].plot(t, chan['ImagA'], color='k')
  ax[1,2].plot(t, chan['ImagB'], color='r')
  ax[1,2].plot(t, chan['ImagC'], color='b')
  ax[1,2].grid()

  ax[1,3].set_title ('Current Angle')
  ax[1,3].set_ylabel ('rad')
  ax[1,3].plot(t, chan['IangA'], color='k')
  ax[1,3].plot(t, chan['IangB'], color='r')
  ax[1,3].plot(t, chan['IangC'], color='b')
  ax[1,3].grid()

  for j in range(4):
    ax[1,j].set_xlabel ('Seconds')

  if file_name:
    plt.savefig(file_name)
  plt.show()

atp_base = 'GFM_v6'
if len(sys.argv) > 1:
  atp_base = sys.argv[1]

channels = h5utils.load_atp_comtrade_channels (atp_base)
make_plot (channels, atp_base)
h5utils.save_atp_channels ('gfm.hdf5', atp_base, channels)
