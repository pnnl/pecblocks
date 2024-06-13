.. _data_from_emt:

Data from EMT
=============

*pecblocks* reads data from HDF5 files that should conform to these requirements:

* Each *case*, i.e., an EMT simulation result or lab event record, should be a separate group in the file, numbered from 0 to *n-1*
* An optional string prefix may be part of the group names, e.g., use *case0* or just *0*. If a prefix is used, it must be uniform among all groups.
* Each channel should conform to the same set of time points at uniform spacing, and of the same length.
* Each channel should be a keyed item within each group.
* The time step should be no more than 2 ms, and preferably smaller.
* The channels must include:

  - *t* for the time
  - *Vd* and *Vq* for the inverter terminal AC voltages (even for single-phase inverters). Treating these as output channels will result in a Thevenin model, using *Id* and *Iq* as input channels.
  - *Id* and *Iq* for the inverter terminal AC voltages (even for single-phase inverters). Treating these as output channels will result in a Norton model, using *Vd* and *Vq* as input channels.

* The channels should include the following as well.  If absent, the model may be trained with constant values or missing inputs, as appropriate:

  - *G* for the solar irradiance
  - *T* for the solar panel temperature
  - *Fc* for the control frequency, from an external controller
  - *Ud* (or *Md*) for the d-axis voltage modulation index, from an external controller
  - *Uq* (or *Mq*) for the q-axis voltage modulation index, from an external controller
  - *Ctl* (or *Ctrl*, *Ramp*, or *Step*) representing an operational mode of the inverter, e.g., 0 for startup, 1 for grid-formed, and 2 for grid-following. These signals could be available from an external controller.
  - *Unb* representing unbalanced operation of the inverter, from an external zero-sequence or negative-sequence protection device. If not used, the trained model will always operate in balanced mode.
  - *GVrms* is desirable for Norton models as a polynomial input feature; it is roughly proportional to the panel power
  - *GIrms* is desirable for Thevenin models as a polynomial input feature; it is roughly proportional to the panel power
  - *Vdc* for the inverter's DC link voltage, which is not coupled to the grid but may be useful as input to an external controller
  - *Idc* for the inverter's DC current, which is not coupled to the grid but may be useful as input to an external controller

* The HDF5 file may include extra channels with no impact on the model training process. The JSON configuration file specifies which input and output channels to use in the model; other channels are ignored.
* The training dataset should encompass the expected range of operation for all input and output channels. Typically, hundreds or thousands of cases are necessary for a good model. The model outputs may deviate substantially from expected behavior if the inputs deviate from the ranges that the model was trained for.
* The training data should initialize smoothly, and settle to stable final values.
* Use  `h5view.py <https://github.com/pnnl/pecblocks/tree/master/examples/data_prep/h5view.py>`_ to verify proper structure of an HDF5 file, before using it for training.
* Use a script like `UCFTrainingPlot.py <https://github.com/pnnl/pecblocks/tree/master/examples/data_prep/UCFTrainingPlot.py>`_ to verify the range of channel values in an HDF5 file, before using it for training.

The EMT solvers and laboratory transient recorders generally do not save 
data in HDF5 format. It's necessary to convert their *native data format* to 
HDF5, and in conjunction, create or process channels to meet the 
requirements listed above. Some of the approaches include:
 
* The EMT simulators (ATP, EMTP, PSCAD) all provide some way of producing IEEE Standard C37.111 (COMTRADE) files. The `Python Comtrade Package <https://pypi.org/project/comtrade/>`_ is then able to load the data for processing by *numpy* or *scipy* functions, then save the data to HDF5 format with *h5py*.
* MATLAB/Simscape saves data in native *mat files*. The Python *scipy* package has a *loadmat* function for these files. Then use *h5py* to save the data to HDF5 format, after any necessary processing.
* Lab data recorders and other data sources may save data in comma-separated value (CSV) files. The Python *numpy* package has a *loadtxt* function for this use case.
* Lab data recorders and other data sources may save data in Microsoft Excel (XLSX) files. The Python *pandas* package has a *read_excel* function for this use case.

The *pecblocks* code will normalize data in the HDF5 file, and optionally 
downsample the data, but it does no other pre-processing. Therefore, 
supplemental inputs like *Vrms*, *GVrms*, *Irms*, and *GIrms* should be 
calculated ahead of time from *Vd*, *Vq*, *Id*, and *Iq*.
 
Experience has shown that *Vrms* and *Irms* are insufficient model inputs 
(or outputs) on their own. The *dq-axis* voltages and currents should be 
provided instead.
 
* The *dq* transformations may be implemented in the EMT model, in which case the *dq* voltages and currents are directly available.
* For lab test records, or EMT simulations without *dq* outputs, the *dq* transformation can be implemented in Python code that adds the necessary HDF5 channels. For an example, see the *simulate_pll* function in `sdi5_prep.py <https://github.com/pnnl/pecblocks/blob/master/examples/data_prep/sdi_prep5.py>`_.
* For single-phase inverters, an orthogonal signal generator may be used to construct the *dq* values. This was implemented within an ATP simulation, following references 18 and 19 of `SysDO 2024 Paper (submitted) <_static/paper.pdf>`_.
* Lab test records, and EMT simulations of detailed switching inverter models, may include significant noise in voltage and current channels. Filtering may be advisable. This might be done as part of the EMT simulation, i.e., by simulating the measurement systems. It might also be done in post-processing. For an example of Butterworth filtering to mitigate measurement noise, see the *my_decimate* function in `sdi5_prep.py <https://github.com/pnnl/pecblocks/blob/master/examples/data_prep/sdi_prep5.py>`_.
* For unbalanced simulations, the *dq* components may be filtered into low-pass and high-pass components, and fit separately. For an example, see `UnbalancedPrep.py <https://github.com/pnnl/pecblocks/blob/master/examples/data_prep/UnbalancedPrep.py>`_.

