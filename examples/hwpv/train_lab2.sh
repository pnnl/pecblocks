#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
#python pv3_training.py ./lab2/lab2_config.json $DATA_PATH/lab2.hdf5
python pv3_metrics.py ./lab2/lab2_config.json $DATA_PATH/lab2.hdf5
