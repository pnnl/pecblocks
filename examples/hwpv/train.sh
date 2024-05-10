#!/bin/bash
if [ ! -d "$1" ]; then
  echo "creating directory $1"
  mkdir $1
fi
python3 pv3_training.py $1_config.json

