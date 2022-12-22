#!/bin/bash
python3 pv3_export.py ./osg4_uvm/osg4_uvm_config.json > osg4_uvm/metrics.txt
python3 pv3_metrics.py ./osg4_uvm/osg4_uvm_config.json /media/sf_src/data/osg4_vdvq.hdf5 >> osg4_uvm/metrics.txt

