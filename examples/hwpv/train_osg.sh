#!/bin/bash
#declare -r DATA_PATH=/media/sf_src/data
declare -r DATA_PATH=$HOME/Documents/data
python3 pv3_training.py ./osg4_uvm/osg4_uvm_config.json $DATA_PATH/osg4_vdvq.hdf5
