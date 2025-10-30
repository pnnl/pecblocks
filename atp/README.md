# atp

This repository contains scripts and Python utilities for use with the Alternative Transients Program (ATP).
Please ask the [European EMTP Users Group](https://www.emtp.org) for information about ATP licensing.
 

## File Directory

These come from the SETO grid-forming inverter controls project. They build on the [pecblocks repository]
(https://github.com/pnnl/pecblocks)
 
 
### ATPDraw Files
 
These are used to create the most recent ATP netlist files, which run by scripting to make the EMT training data.
 
- GFM_v8.acp, GFM_v8_test.acp and GFM_v8_vsHWPV.acp for the three-phase inverter
- pv1_osg.acp for the single-phase inverter
- _developmental_ subdirectory contains a few dozen files that were used in developing sub-systems of the main ATP models.

### Python Scripts

Some of these include, as of 2021:

- atp_loop_hw.py; creates parameter sets in GFM_v6.prm to run and process G, T, and balanced Rload variations
- atp_loop_hw_1ph.py; creates parameter sets in GFM_v6_1ph.prm to run and process G, T, and unbalanced Rload variations
- comtrade.py; forked from https://github.com/dparrini/python-comtrade with patches, to support COMTRADE 1991-2013
- ComtradeTrainingPlot.py; plots a single ATP simulation case, processed to phasor quantities for HW fitting
- h5utils.py; functions to read and filter channels from COMTRADE files, save in HDF5 format
- h2view.py; prints the group structure of an HDF5 file
- HDF5TrainingPlot.py; plots a set of ATP simulation cases, processed to phasor quantities for HW fitting
- hw_model_block.py; Python script that produces _*.mod_ from _models.json_
- hw_norton_model.py; Python script that produces _GIDSRC.mod_ from _models.json_

### ATP Test Files

Some of these include, as of 2021:

- atp_loop_hw.bat; runs atp_loop_hw.py on GFM_v6.atp
- atp_loop_hw_1ph.bat; runs atp_loop_hw_1ph.py on GFM_v6_1ph.atp
- clean.bat; batch file to clean up outputs and log files from an ATP simulation
- default_model.acp; ATPDraw model for a simple MODELS example; run this and look at default_model.atp for a template
- runtp.bat; batch file to run the regular ATP
- GFM_v6.atp; CERTS average GFM model, with MPPT, modified to vary G, T, and balanced Rload from included GFM_v6.prm file
- GFM_v6_1ph.atp; GFM_v6.atp, modified to run single-phase load steps from included GFM_v6_1ph.prm file
- runtpgig.bat; batch file to run the upsized ATP
- test_models.atp; a test harness for running SISO _\*.mod_ files
- test_norton.atp; a test harness for running GIDSRC.mod, the controlled Norton source

### Archived Models

- GIDSRC.mod; sample controlled Norton source HW model
- GTOIDC.mod; sample FGF block diagram model for irradiance to DC current
- GTOVDC.mod; sample FGF block diagram model for irradiance to DC voltage
- TTOIDC.mod; sample FGF block diagram model for temperature to DC current
- TTOVDC.mod; sample FGF block diagram model for temperature to DC voltage
- VTOIDC.mod; sample FGF block diagram model for DC voltage to DC current
- VTOPAC.mod; sample FGF block diagram model for DC voltage to AC real power
- VTOQAC.mod; sample FGF block diagram model for DC voltage to AC reactive power

### Archived Trained Model Coefficients

_models.json_ contains the attributes for illustrative single-input, single-output (SISO) generalized block diagram models. The structure of this file is:

- top-level model entries are keyed on the model name, which is limited to 6 characters for ATP
    - second-level _name_ attribute should match the top-level key, which is limited to 6 characters for ATP
    - second-level _type_ attribute indicates the block structure, e.g., "F1+G1+F2" or "Fnl+G2"
    - second-level _G\*_ attribute indicates a linear block; this key should match the block position in _type_. There are zero or more such blocks. The discrete time step used for fitting is 20 ms.
        - third-level _denominator_ attribute is an array of coefficients, of length equal to polynomial order, beginning with z-1. The implied z0 coefficient is always one.
        - third-level _numerator_ attribute is an array of coefficients, of length equal to polynomial order, beginning with z-1. The implied z0 coefficient is always zero.
        - third_level _n_k_ attribute is an integer number of delay steps, i.e., number of _t_step_ delays in the output.  Zero or more. _t_step_ is 20 ms in this case.
    - second-level _F\*_ attribute indicates a nonlinear block; this key should match the block position in _type_. The activation function is _tanh_. There are zero or more such blocks.
        - third_level _n0b_ attribute is an array of input layer bias coefficients
        - third_level _n0w_ attribute is an array of input layer weight coefficients
        - third_level _n2b_ attribute is an array of output layer bias coefficients
        - third_level _n2w_ attribute is an array of output layer weight coefficients

## HW Block Example Steps

1. ```python hw_model_block.py``` to create ```*.mod``` from ```models.json```
2. ```runtp test_models.atp``` to run ATP on the ```*.mod``` models
3. Use ```ATP/PlotXY``` from the Windows Start menu to plot the output file ```test_models.pl4```

## HW Training Example Steps

Balanced 3-phase startup, then a balanced change to feasible operating point:

1. ```atp_loop_hw.bat```
2. ```python ComtradeTrainingPlot.py gfm_v6``` for the last case plot
3. ```python HDF5TrainingPlot.py looped.hdf5``` for all training waveforms superimposed

Balanced 3-phase startup, then an **unbalanced** change to feasible operating point:

1. ```atp_loop_hw_1ph.bat```
2. ```python ComtradeTrainingPlot.py gfm_v6_1ph``` for the last case plot
3. ```python HDF5TrainingPlot.py looped_1ph.hdf5``` for all training waveforms superimposed

## HW Norton Source Example Steps

1. ```python hw_norton_model.py``` to create ```GIDSRC.mod``` from ```models.json```
2. ```runtp test_norton.atp``` to run ATP on the ```GIDSRC.mod``` models
3. Use ```ATP/PlotXY``` from the Windows Start menu to plot the output file ```test_norton.pl4```

## Notice

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

    PACIFIC NORTHWEST NATIONAL LABORATORY
                operated by
                 BATTELLE
                 for the
     UNITED STATES DEPARTMENT OF ENERGY
      under Contract DE-AC05-76RL01830

Copyright 2021-25, Battelle Memorial Institute