#!/bin/bash
python3 pv3_export.py $1_config.json > $1/metrics.txt
python3 pv3_metrics.py $1_config.json >> $1/metrics.txt

