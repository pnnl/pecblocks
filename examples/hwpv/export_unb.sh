#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_export.py unb3/unb3_config.json > unb3/metrics.txt
python pv3_metrics.py unb3/unb3_config.json $DATA_PATH/unb3.hdf5 >> unb3/metrics.txt

