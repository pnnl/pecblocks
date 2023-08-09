#!/bin/bash
declare -r DATA_PATH=$HOME/Documents/data
python pv3_export.py ./big3/big3_config.json > big3/metrics.txt
python pv3_metrics.py ./big3/big3_config.json $DATA_PATH/big3.hdf5 >> big3/metrics.txt

