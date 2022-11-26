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
import h5py
import matplotlib.pyplot as plt
import glob

# these are for the September dataset
plot_defs = [
  {'row':0, 'col':0, 'tags':['G'], 'title':'Irradiance', 'ylabel':'W/m2'},
  {'row':0, 'col':1, 'tags':['T'], 'title':'Temperature', 'ylabel':'deg C'},
  {'row':0, 'col':2, 'tags':['Ctl'], 'title':'Control Mode', 'ylabel':'pu'},
  {'row':0, 'col':3, 'tags':['Fc'], 'title':'Control Frequency', 'ylabel':'Hz'},

  {'row':1, 'col':0, 'tags':['GVrms'], 'title':'Polynomial Feature', 'ylabel':''},
  {'row':1, 'col':1, 'tags':['Md','Mq'], 'title':'Modulation Indices', 'ylabel':'pu'},
  {'row':1, 'col':2, 'tags':['Vdc','Vrms'], 'title':'DC/RMS Voltages', 'ylabel':'V'},
  {'row':1, 'col':3, 'tags':['Idc'], 'title':'DC Current', 'ylabel':'A'},

  {'row':2, 'col':0, 'tags':['Id','Iq'], 'title':'PCC Currents', 'ylabel':'A'},
  {'row':2, 'col':1, 'tags':['Isd','Isq'], 'title':'Inverter Currents', 'ylabel':'A'},
  {'row':2, 'col':2, 'tags':['Vbd','Vbq'], 'title':'Voltages at B?', 'ylabel':'V'},
  {'row':2, 'col':3, 'tags':['Vod','Voq'], 'title':'Voltages at O?', 'ylabel':'V'},
]

# these are for the November dataset
plot_defs = [
  {'row':0, 'col':0, 'tags':['G'], 'title':'Irradiance', 'ylabel':'W/m2'},
  {'row':0, 'col':1, 'tags':['T'], 'title':'Temperature', 'ylabel':'deg C'},
  {'row':0, 'col':2, 'tags':['Ctl'], 'title':'Control Mode', 'ylabel':'pu'},
  {'row':0, 'col':3, 'tags':['Fc'], 'title':'Control Frequency', 'ylabel':'Hz'},

  {'row':1, 'col':0, 'tags':['GVrms'], 'title':'Polynomial Feature', 'ylabel':''},
  {'row':1, 'col':1, 'tags':['Md1','Mq1'], 'title':'Modulation Indices', 'ylabel':'pu'},
  {'row':1, 'col':2, 'tags':['Vdc'], 'title':'DC Voltage', 'ylabel':'V'},
  {'row':1, 'col':3, 'tags':['Idc'], 'title':'DC Current', 'ylabel':'A'},

  {'row':2, 'col':0, 'tags':['Id'], 'title':'PCC Currents', 'ylabel':'A'},
  {'row':2, 'col':1, 'tags':['Iq'], 'title':'Inverter Currents', 'ylabel':'A'},
  {'row':2, 'col':2, 'tags':['Vod','Vrms'], 'title':'Vd and Vrms', 'ylabel':'V'},
  {'row':2, 'col':3, 'tags':['Voq'], 'title':'Vq', 'ylabel':'V'},
]

nrows = 3
ncols = 4

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
  if bSummarize:
    df.info()
    print ('Column                         Min           Max          Mean')
    for lbl, data in df.iteritems():
      print ('{:20s} {:13.5f} {:13.5f} {:13.5f}'.format (lbl, data.min(), data.max(), data.mean()))
  return df, casenumber

def plot_dataframe (df, casenumber):
  plt.rcParams['savefig.directory'] = os.getcwd()
  fig, ax = plt.subplots (nrows, ncols, sharex = 'col', figsize=(18,8), constrained_layout=True)
  fig.suptitle ('Simulation Results from Case {:s}'.format(casenumber))
  t = df.index
  for plot in plot_defs:
    plt_ax = ax[plot['row'], plot['col']]
    for tag in plot['tags']:
      plt_ax.plot (t, df[tag], label=tag)
    plt_ax.set_title (plot['title'])
    plt_ax.set_ylabel (plot['ylabel'])
    plt_ax.grid()
    plt_ax.legend (loc='best')
  plt.show()
  plt.close()

if __name__ == '__main__':
  inputfile = 'case500.xlsx'
  outputfile = 'test.hdf5'
  if len(sys.argv) > 1:
    inputfile = sys.argv[1]
    if len(sys.argv) > 2:
      outputfile = sys.argv[2]

  mode = 'w'
  if '*' in inputfile:
    files = glob.glob (inputfile)
    print ('Writing {:d} xlsx files to {:s}'.format (len(files), outputfile))
    for fname in files:
      df, casenumber = read_one_xlsx (fname, bSummarize=False)
      saveas_comtrade_channels (outputfile, casenumber, df, mode=mode)
      mode = 'a'
  else:
    df, casenumber = read_one_xlsx (inputfile, bSummarize=True)
    saveas_comtrade_channels (outputfile, casenumber, df, mode=mode)
    plot_dataframe (df, casenumber)


