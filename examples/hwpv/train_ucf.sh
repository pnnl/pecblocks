#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_training.py ./ucf2ac/ucf2ac_config.json $DATA_PATH/ucf2.hdf5
