# Copyright (C) 2021 Battelle Memorial Institute

from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import time
import pecblocks.util

training_sets = [{'Title':'DC Training to Temperature', 
                  'Path':'./data/emt_temperature.zip',
                  'PlotFile':'TemperatureTraining.pdf',
                  'tmin':3.5,
                  'tmax':5.0,
                  'u1':{'tag':'MODELS TMPOUT_I-branch', 'name':'Temperature'},
                  'u2':{'tag':'MODELS IRROUT_I-branch', 'name':'Irradiance'},
                  'y1':{'tag':'VCAP_V-branch', 'name':'DC Voltage'},
                  'y2':{'tag':'DCND_I-branch', 'name':'DC Current'}},
                 {'Title':'DC Training to Irradiance', 
                  'Path':'./data/emt_irradiance.zip',
                  'PlotFile':'IrradianceTraining.pdf',
                  'tmin':3.5,
                  'tmax':5.0,
                  'u1':{'tag':'MODELS TMPOUT_I-branch', 'name':'Temperature'},
                  'u2':{'tag':'MODELS IRROUT_I-branch', 'name':'Irradiance'},
                  'y1':{'tag':'VCAP_V-branch', 'name':'DC Voltage'},
                  'y2':{'tag':'DCND_I-branch', 'name':'DC Current'}},
                 {'Title':'AC Training to DC Voltage', 
                  'Path':'./data/average_irradiance.zip',
                  'PlotFile':'VdcTraining.pdf',
                  'tmin':2.5,
                  'tmax':4.0,
                  'u1':{'tag':'VDC_V-node', 'name':'DC Voltage'},
                  'u2':{'tag':'XX0004 DCND_I-branch', 'name':'DC Current'},
                  'y1':{'tag':'MODELS PAC_I-branch', 'name':'AC Real Power'},
                  'y2':{'tag':'MODELS QAC_I-branch', 'name':'AC Reactive Power'}}
                ]

def plot_training_set (row):
    decimate = 1
    t_step = 0.00002
    n_skip = 0

    data_path = row['Path']

    df_training = pecblocks.util.read_csv_files(data_path, pattern=".csv")

    df1 = df_training[(df_training['time']>row['tmin']) & (df_training['time']<row['tmax'])]
#    print ('window df1', df1.describe())
    print ('df1 columns', df1.columns)
#    print ('df1 index', df1.index)
    print ('df1 shape', df1.shape)

    df_u1 = df1.filter(like=row['u1']['tag'], axis=1)
    df_u2 = df1.filter(like=row['u2']['tag'], axis=1)
    df_y1 = df1.filter(like=row['y1']['tag'], axis=1)
    df_y2 = df1.filter(like=row['y2']['tag'], axis=1)
    print ('df_y2 shape filtered', df_y2.shape)

    df_y1 = df_y1.sum(axis=1)
    df_u1 = df_u1.sum(axis=1)
    df_y2 = df_y2.sum(axis=1)
    df_u2 = df_u2.sum(axis=1)
    print ('df_y2 shape summed', df_y2.shape)

    # Extract data
    y1 = np.transpose(df_y1.to_numpy(dtype=np.float32))# /1000.0  # batch, time, channel
    u1 = np.transpose(df_u1.to_numpy(dtype=np.float32))# /1000.0 # 1.0  # normalization of T or G
    y2 = np.transpose(df_y2.to_numpy(dtype=np.float32))# /1000.0  # batch, time, channel
    u2 = np.transpose(df_u2.to_numpy(dtype=np.float32))# /1000.0 # 1.0  # normalization of T or G
    print ('y2 shape', y2.shape)
    t = np.arange(0,  (y1.size)*t_step, t_step, dtype=np.float32)

    y1 = y1.reshape(y1.size,1)
    u1 = u1.reshape(u1.size,1)
    y2 = y2.reshape(y2.size,1)
    u2 = u2.reshape(u2.size,1)
    t = t.reshape(t.size,1)

    n_fit = y1.size-1

    print ('n_fit={:d}, decimate={:d}'.format (n_fit, decimate))

    # In[Fit data, slicing arrays to customize the sampling rate]
    y1_fit = y1[0:n_fit:decimate]
    u1_fit = u1[0:n_fit:decimate]
    y2_fit = y2[0:n_fit:decimate]
    u2_fit = u2[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    lsize = 10
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=lsize)
    plt.rc('ytick', labelsize=lsize)
    plt.rc('axes', labelsize=lsize)
    plt.rc('legend', fontsize=lsize)
    pWidth = 6.0
    pHeight = pWidth / 1.618
    bwidth = 0.5
    fig, ax = plt.subplots(2, 2, figsize=(pWidth, pHeight), constrained_layout=True)
    plt.suptitle (row['Title'])
    ax[0,0].set_title ('u1: {:s}'.format (row['u1']['name']), fontsize=lsize)
    ax[0,0].plot (t_fit[n_skip:], u1_fit[n_skip:], 'k')
    ax[1,0].set_title ('u2: {:s}'.format (row['u2']['name']), fontsize=lsize)
    ax[1,0].plot (t_fit[n_skip:], u2_fit[n_skip:], 'r')
    ax[0,1].set_title ('y1: {:s}'.format (row['y1']['name']), fontsize=lsize)
    ax[0,1].plot (t_fit[n_skip:], y1_fit[n_skip:], 'g')
    ax[1,1].set_title ('y2: {:s}'.format (row['y2']['name']), fontsize=lsize)
    ax[1,1].plot (t_fit[n_skip:], y2_fit[n_skip:], 'b')

    plot_folder = './models/training/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(plot_folder + row['PlotFile'])
    plt.show()
    plt.close()

if __name__ == '__main__':
    for row in training_sets:
        plot_training_set (row)
#        quit()

