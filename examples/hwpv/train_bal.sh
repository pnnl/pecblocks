#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_training.py ./big3/big3_config.json $DATA_PATH/big3.hdf5
