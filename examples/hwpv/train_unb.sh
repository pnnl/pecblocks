#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_training.py ./unb3/unb3_config.json $DATA_PATH/unb3.hdf5
