# copyright 2021-2022 Battelle Memorial Institute
# reads a spreadsheet of Simscape output, plots channels and converts to HDF5
#  arg1: input file name, format like case###.xlsx, default case500.xlsx, *.xlsx to glob
#  arg2: output file name, default test.hdf5
#  example: python xlsx_to_hdf5.py newcase.xlsx new.hdf5
#
# This script reads the XLSX file into a Pandas Dataframe, because
# Pandas has a one-step function to import XLSX files. However, the
# HWPV training scripts do not use Pandas. Therefore, the saved HDF5
# file uses the same schema as for COMTRADE analog channels, which
# are compatible with HWPV training scripts.

# this next line would save the Pandas Dataframe, which is not suited for HWPV fitting
#  df.to_hdf(outputfile, key='Test', mode='w')
#  df.plot(subplots=True)

import sys
import os
import pandas as pd
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

plot_defs = [
  {'row':0, 'col':0, 'tags':['G'], 'title':'G', 'ylabel':'W/m2'},
#  {'row':0, 'col':1, 'tags':['T'], 'title':'Temperature', 'ylabel':'deg C'},
  {'row':0, 'col':1, 'tags':['Ctl'], 'title':'Ctl', 'ylabel':'pu'},
  {'row':0, 'col':2, 'tags':['Fc'], 'title':'Fc', 'ylabel':'Hz'},
  {'row':0, 'col':3, 'tags':['Md'], 'title':'Md', 'ylabel':'pu'},
  {'row':0, 'col':4, 'tags':['Mq'], 'title':'Mq', 'ylabel':'pu'},

  {'row':1, 'col':0, 'tags':['Vd'], 'title':'Vd', 'ylabel':'V'},
  {'row':1, 'col':1, 'tags':['Vq'], 'title':'Vq', 'ylabel':'V'},
  {'row':1, 'col':2, 'tags':['Vrms'], 'title':'Vrms', 'ylabel':'V'},
  {'row':1, 'col':3, 'tags':['GVrms'], 'title':'GVrms', 'ylabel':''},
  {'row':1, 'col':4, 'tags':['P'], 'title':'P', 'ylabel':'kW'},

  {'row':2, 'col':0, 'tags':['Vdc'], 'title':'Vdc', 'ylabel':'V'},
  {'row':2, 'col':1, 'tags':['Idc'], 'title':'Idc', 'ylabel':'A'},
  {'row':2, 'col':2, 'tags':['Id'], 'title':'Id', 'ylabel':'A'},
  {'row':2, 'col':3, 'tags':['Iq'], 'title':'Iq', 'ylabel':'A'},
  {'row':2, 'col':4, 'tags':['Q'], 'title':'Q', 'ylabel':'kVAR'},
]

nrows = 3
ncols = 5

# to save multiple cases in the same HDF5, the first call should
# have mode='w', and subsequent calls should have mode='a'
def saveas_comtrade_channels (filename, groupname, df, mode='w'):
  f = h5py.File (filename, mode)
  grp = f.create_group (groupname)
  grp.create_dataset ('t', data=df.index, compression='gzip')
  for lbl, data in df.iteritems():
    grp.create_dataset (lbl, data=data, compression='gzip')
  f.close()

def read_one_xlsx (inputfile, bSummarize=False):
  casenumber = os.path.basename(inputfile).rstrip('.xlsx').lstrip('case')
  if not bSummarize:
    print ('Processing {:s} as case {:s}'.format(inputfile, casenumber))
  df = pd.read_excel (inputfile)
  df.set_index ('t', inplace=True)
  df.drop (columns=['Vrms','GVrms'], inplace=True) # keeping Isd and Isq for now
  df.rename (columns={'Md1':'Md','Mq1':'Mq','Vod':'Vd','Voq':'Vq'}, inplace=True)
  df['Vrms'] = math.sqrt(1.5)*np.sqrt(df['Vd']*df['Vd'] + df['Vq']*df['Vq'])
  df['GVrms'] = 0.001 * df['G'] * df['Vrms']
  df['P'] = 0.0015 * (df['Vd']*df['Id'] + df['Vq']*df['Iq'])
  df['Q'] = 0.0015 * (df['Vq']*df['Id'] - df['Vd']*df['Iq'])
  if bSummarize:
    df.info()
    print ('Column                         Min           Max          Mean')
    for lbl, data in df.iteritems():
      print ('{:20s} {:13.5f} {:13.5f} {:13.5f}'.format (lbl, data.min(), data.max(), data.mean()))
  return df, casenumber

def add_dataframe_plot (ax, df, casenumber):
  t = df.index
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    for tag in plot['tags']:
      plt_ax.plot (t, df[tag], label=tag)

def start_plot(n):
  plt.rcParams['savefig.directory'] = os.getcwd()
  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Simulation Results from {:d} cases'.format(n))
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])
    plt_ax.grid()
#    plt_ax.legend (loc='best')
  return ax
  
def finish_plot():  
  plt.show()
  plt.close()

if __name__ == '__main__':
  outputfile = 'new.hdf5'
  mode = 'w'
  n = 5
  ax = start_plot (n)
  for i in range(n):
    inputfile = 'case{:d}.xlsx'.format(i)
    df, casenumber = read_one_xlsx (inputfile, bSummarize=True)
    add_dataframe_plot (ax, df, casenumber)
    saveas_comtrade_channels (outputfile, casenumber, df, mode=mode)
    mode = 'a'
  finish_plot()
