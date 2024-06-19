.. role:: math(raw)
   :format: html latex
..

Examples
========

Source files are at https://github.com/pnnl/pecblocks/tree/master/examples/hwpv.
Date files are at the `Harvard Dataverse <https://doi.org/10.7910/DVN/PWW981>`_.
Results from these examples are discussed in `SysDO 2024 Paper (submitted) <_static/paper.pdf>`_

To create their own examples, users will need to provide:

* A configuration file, as described in :ref:`schema`
* A training dataset from laboratory tests or EMT simulations, as described in :ref:`data_from_emt`

Training a Model
----------------

The following script trains a model on Windows. It's called with one 
argument, the root of the configuration file, e.g., `call train.bat osg4`, 
assuming that `osg4_config.json` has been created. The directory `osg4` 
will be created if necessary, but not erased. The last line plots training 
losses to a saved PDF, but not to the screen so that a calling script may 
continue without user acknowledgement. 

.. literalinclude:: ../examples/hwpv/train.bat

The followng script is for Linux and Mac OS X, called like `./train.sh osg4`

.. literalinclude:: ../examples/hwpv/train.sh

The following Python code supervises training a model using *pecblocks*.

* Line 12 disables the output of mean absolute error (MAE), because root mean square error (RMSE) is always more suitable for HWPV applications.
* Line 13 disables the plot of losses at the end of a training run. This plot waits for user input, which prevents a batch file from completing several training runs. To plot the losses afterward, invoke `python loss_plot.py osg4`
* Lines 15-30 open the required configuration file.
* Lines 32-38 comprise the main usage of *pecblocks* API

  - Line 32 initializes a *model* instance from the JSON configuration file
  - Line 33 loads the training data set, which may take several seconds or longer
  - Line 34 normalizes the training data to a range 0..1 on each channel, and saves the normalization factors
  - Line 35 prepares the *model* for training (and evaluation)
  - Line 36 trains new model coefficients, which may take several minutes or longer. Progress updates are written to the console.
  - Line 37 saves the most recent trained model coefficients. (Note: these coefficients are saved periodically during training. The set of coefficients with lowest fitting loss is also saved as needed in each epoch.)
  - Line 38 summarizes the training errors for each output channel, over all cases or events in the training data set. More detail is available in the next steps. The RMSE is most relevant to HWPV applications.

* Lines 40-59 provide summary output of the training run. This information is also availble later.
* Lines 61-73 plot the losses vs. training epoch. Use of *loss_plot.py* is now preferred.

.. literalinclude:: ../examples/hwpv/pv3_training.py
  :linenos:

Exporting a Model
-----------------

The following script exports a trained model for simulation, and its 
accuracy metrics on Windows. It's called with one argument, the root of 
the configuration file, e.g., `call export.bat osg4`, assuming that 
`osg4_config.json` has been created. Furthermore, the directory `osg4` 
should have been created and populated by training the model. 

.. literalinclude:: ../examples/hwpv/export.bat

The followng script is for Linux and Mac OS X, called like `./export.sh osg4`

.. literalinclude:: ../examples/hwpv/export.sh

The following Python code exports (post-processes) a trained model for 
simulation using *pecblocks*.
 
* Lines 10-27 open the required configuration file
* Lines 29-34 comprise the main usage of *pecblocks* API

  - Line 29 initializes a *model* instance from the JSON configuration file, same as for training.
  - Line 30 loads the normalization factors from training. To export the model, it's not necessary to load the training data set.
  - Line 31 prepares the *model* for export
  - Line 32 loads the trained model coefficients from binary *pkl* files, which are not human-readable.
  - Line 33 exports the trained model coefficients in human-readable JSON file. During this process, and *s-domain* version of the model is prepared for simulation at variable time step. The JSON file is readable by other applications in a variety of languages, including Python, C/C++, and MATLAB.
  - Line 34 checks for stability of the exported *H1* poles. If there are any warnings about unstable poles, do not use this model for simulation. Specifying the *stable2ndx* type for *H1* in a re-trained model should avoid this issue.

.. literalinclude:: ../examples/hwpv/pv3_export.py
  :linenos:

Model Metrics
-------------

The following Python code summarizes metrics of a trained model using *pecblocks*.

* Line 11 disables the output of mean absolute error (MAE), because root mean square error (RMSE) is always more suitable for HWPV applications.
* Lines 13-28 open the required configuration file.
* Lines 30-35 comprise the main usage of *pecblocks* API

  - Line 30 initializes a *model* instance from the JSON configuration file, same as for training or exporting.
  - Line 31 loads the training data set, which may take several seconds or longer
  - Line 32 loads and applies normalization factors established during training.
  - Line 33 prepares the *model* for evaluation (and training)
  - Line 34 loads the trained model coefficients from binary *pkl* files, which are not human-readable.
  - Line 35 summarizes the training errors for each output channel, individually by case or event, and over all cases.

* Lines 36-76 provide summary output, which includes:

  - The RMSE for each output channel and each case.
  - Identifying the case number that produced the highest RMSE for each output channel.
  - The number of cases in which the RMSE exceeded 0.05 per-unit, by output channel. This should be considered in judging whether the trained model is acceptable for use.
  - The total RMSE over all cases, for each output channel. These values are the same as output at the end of the training run. However, some cases will have higher RMSE values. Furthermore, some cases may exceed 0.05 RMSE, even when the total RMSE does not exceed 0.05 perunit.

.. literalinclude:: ../examples/hwpv/pv3_metrics.py
  :linenos:

Running a Model
---------------

Once the model has been exported, it can be run in a dynamic or EMT
simulation with forward evaluation of the HWPV blocks. Two examples
are described below. In addition, the models have been run in MATLAB/Simscape
and ATP, with a Cigre/IEEE "real code" DLL implementation in progress.

Co-simulation using HELICS
^^^^^^^^^^^^^^^^^^^^^^^^^^

This example runs in the **z** domain, with a fixed time step that must
match the time step used to train the model. It runs as a co-simulation,
with two Python federates communicating over the HELICS interface. One federate
plays the role of the grid, while the other plays the role of a HWPV inverter model.
In a more realistic use case, the grid federate might be a commercial dynamics
or EMT simulator that already supports HELICS.

* See `IEEE Access Paper <https://doi.org/10.1109/ACCESS.2024.3363615>`_ for more information about HELICS.
* The example files and a description are at `<https://github.com/pnnl/pecblocks/tree/master/examples/helics>`_

Standalone Simulation using Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example runs in the **s** domain using a Backward Euler integration 
of the *H1* block. The time step is longer than the step used to train
the model. Although not shown in this example, the Backward Euler method
allows for variable time step during a simulation.

* The example files and a description are at `<https://github.com/pnnl/pecblocks/tree/master/examples/pi>`_
* The file `hwpv_pi.py` plays the role of a grid federate, applying the test stimuli, calculating voltage from current at the PCC, and calling the HWPV model at each time step
* The file `hwpv_evaluator.py` implements the HWPV block model using *numpy*, but not *pecblocks*, *dynonet*, or *torch*. It can run on a Raspberry Pi or larger computer.
* The evaluator supports the z domain, the s domain with Forward Euler (less accurate and possibly unstable), and the s domain with Backward Euler (preferred).

