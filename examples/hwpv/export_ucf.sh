#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_export.py ./ucf2ac/ucf2ac_config.json > ucf2ac/metrics.txt
python pv3_metrics.py ./ucf2ac/ucf2ac_config.json $DATA_PATH/ucf2.hdf5 >> ucf2ac/metrics.txt

