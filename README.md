# pecblocks

Power Electronic Converter Blocks (pecblocks) use the output of detailed electromagnetic transient (EMT) simulations to produce generalized block
diagram models of power electronic systems. The process uses deep learning with customized block architectures. The
outputs are reduced-order models that meet specified accuracy requirements, while providing important advantages over
the original EMT models:

- Converter Block models have fewer nodes and can take longer time steps, resulting in shorter simulation times
- Converter Block models are continuously differentiable, making them compatible with control design methods

Models will run in both time domain and frequency domain. The scope includes not only the power electronic converter,
but also the photovoltaic (PV) array, maximum power point tracking (MPPT), phase-lock loop (PLL) control, output filter
circuits, battery storage if present, etc. The applications include but may not be limited to solar power inverters, energy
storage converters, motor drives, and other power electronics equipment.

## Panels-to-Grid MIMO Test Case

In the directory _examples/hwpv_, a multiple-input, multiple-output HW model has been
fitted to the PV panels, MPPT, DC/DC, inverter PWM switching, LCL output filter, and RMS measurements.
It was based on ATP simulations with five inputs, _G_, _T_, _Ud_, _Fc_, and _Vrms_. A sixth input is
created as a polynomial feature, defined as _GVrms_. The outputs are _Idc_, _Vdc_, and _Irms_. A
seventh input is added to indicate the inverter control mode; 0=starting, 1=grid formed, 2=grid following.

To run this example, comparing ATP to model output:

    python pv1_import.py

To run the example LCL filter:

    python pv1_lcl.py

### Example Results

![Block Diagram](/examples/hwpv/BlockDiagram.png)

**HW Block Diagram and Normalized RMS Error**

![H1_0_0](/examples/hwpv/H1_0_0.png)
![H1_0_1](/examples/hwpv/H1_0_1.png)
![H1_0_2](/examples/hwpv/H1_0_2.png)
![H1_1_0](/examples/hwpv/H1_1_0.png)
![H1_1_1](/examples/hwpv/H1_1_1.png)
![H1_1_2](/examples/hwpv/H1_1_2.png)
![H1_2_0](/examples/hwpv/H1_2_0.png)
![H1_2_1](/examples/hwpv/H1_2_1.png)
![H1_2_2](/examples/hwpv/H1_2_2.png)

**Bode Plots of the MIMO H1 Transfer Function**

The figure below shows an 8-second simulation of inverter startup, followed
by sequential disturbances in the weather, control variables, and grid resistance.
Inputs appear on the top row. The AC RMS voltage, _Vrms_, comes from an electrical
simulation in the Alternative Transients Program (ATP). In the infinite impulse
response (IIR) simulation from the trained model, _Vrms_ is not available directly,
but assumed to be _Rgrid_*_Irms_, where _Rgrid_ is pre-defined for the IIR simulation
and _Irms_ is an IIR output variable. Hence, there is a lag of one time step, 1 ms,
in _Vrms_ during the IIR simulation. Differences in the ATP and IIR Vrms are partly
responsible for differences in the ATP and IIR output variables, _Vdc_, _Idc_, and _Irms_.

| Variable | RMSE  |  Mean  |  Rel. Error |
|:---      |   ---:|    ---:|         ---:|
|Vdc  | 34.4120 | 365.9086 | 9.40%| 
|Idc  | 0.4247  | 27.2002  | 1.56%| 
|Irms | 6.5667  | 45.5142  | 14.43%|
 
Notes to investigate:

- The output variables do not start at zero output. Bias terms in F1 and F2 may cause this, regardless of any zeroed initial conditions on H1, or adjustments to the normalization factors. A back-initialization may fix the problem.
- There is a change in control mode from "startup" to "grid formed" between 2.0 and 2.1 seconds. This disturbs the output before the first actual disturbance, in G, from 2.4 to 2.5 seconds.
- The entire simulation could be repeated with control mode constant at 0, and again with control model constant at 1.
- The discrete-time IIR filter should be replaced with a continuous-time transfer function.

![IIR_Sim](/examples/hwpv/IIR_Filter_Simulation.png)

**System Simulation Using Infinite Impulse Response Filters in Discrete Time Steps**

The figure below shows the AC output variables with an LCL output filter. Because the
grid impedance is purely resistive, the angle reference at the LCL output is zero
degrees for the voltage and current, _Vc_ and _Ic_, respectively. The phasor voltage
and current behind the filter, _Vs_ and _Is_, respectively, are calculated by complex
arithmetic because the LCL circuit is completely determined. The LCL complex impedances
are updated with _Fc_ at each time step. In this example, _Fc_ changes from 60 Hz to 63 Hz
at approximately 4.5 seconds. There is a voltage drop through the LCL filter, from _Vs_ to
_Vc_, with a small corresponding increase in the output current, _Ic_.

| Case | Lf [mH] | Cf [uF] | Lc [mH] |
|:---  |     ---:|     ---:|     ---:|
|1-phs |  2.0000 |    20.0 |  0.4000 | 
|3-phs |  0.0610 |    19.1 |  0.0367 | 

The 3-phase circuit has parallel damping resistors, _Rf_=91.5 and _Rc_=55.05, providing
a damping factor of 7.5 for each inductor. The _Cf_ has a series resistance of 0.01.

![LCL_Sim](/examples/hwpv/LCL_Simulation.png)

**System Simulation with LCL Output Filter**

### File Directory

The Python files currently used in this example are:

- _pv1_poly.py_ implements a multi-channel Hammerstein-Wiener architecture.
- _pv1_import.py_ reads the model for time-step simulation, produces Bode plots, and compares IIR simulation to ATP simulation
- _pv1_lcl.py_ runs the same case as _pv1_import.py_, with an LCL output filter and plotting the LCL input and output variables.

A sample trained model is provided in _models/pv1_fhf_poly.json_, which contains the following in a readable text format.

- There is only one top-level entry
    - second-level _name_ attribute is limited to 6 characters for ATP
    - second-level _type_ attribute indicates the block structure, e.g., "F1+G1+F2"
    - second-level _t_step_ attribute is the discretization time step
    - second-level _normfacs_ attribute contains the channel scaling factors
        - the key is a column name
        - _offset_ is the channel mean, in physical units
        - _scale_ is the channel range, in physical units
    - second-level _lr_ came from _pv1_config.json_ as described above
    - second-level _num_iter_ came from _pv1_config.json_ as described above
    - second-level _print_freq_ came from _pv1_config.json_ as described above
    - second-level _batch_size_ came from _pv1_config.json_ as described above
    - second-level _n_skip_ came from _pv1_config.json_ as described above
    - second-level _n_trunc_ came from _pv1_config.json_ as described above
    - second-level _n_dec_ came from _pv1_config.json_ as described above
    - second-level _na_ came from _pv1_config.json_ as described above
    - second-level _nb_ came from _pv1_config.json_ as described above
    - second-level _nk_ came from _pv1_config.json_ as described above
    - second-level _activation_ came from _pv1_config.json_ as described above
    - second-level _nh1_ came from _pv1_config.json_ as described above
    - second-level _nh2_ came from _pv1_config.json_ as described above
    - second-level _H*_ attribute indicates a linear block; this key should match the block position in _type_. There could be zero or more such blocks, but currently one. The discrete time step used for fitting is 1 ms.
        - third_level _n_in_ attribute is the number of input channels, should match the overall number of HW inputs
        - third_level _n_out_ attribute is the number of output channels, should match the overall number of HW outputs
        - third_level _n_k_ attribute is an integer number of delay steps, i.e., number of _t_step_ delays in the output.  Zero or more.
        - third_level _n_a_ attribute is the number of denominator coefficients
        - third_level _n_b_ attribute is the number of numerator coefficients
        - third-level _a_i_j_ attributes are arrays of denominator coefficients, of length equal to polynomial order, beginning with z-1. The implied z0 coefficient is always one.
            - in the attribute name, _i_ is the output channel number, ranging from 0 to _n_out_ - 1
            - in the attribute name, _j_ is the input channel number, ranging from 0 to _n_in_ - 1
        - third-level _b_i_j_ attributes are arrays of numerator coefficients, of length equal to polynomial order, beginning with z-1. The implied z0 coefficient is always zero.
            - in the attribute name, _i_ is the output channel number, ranging from 0 to _n_out_ - 1
            - in the attribute name, _j_ is the input channel number, ranging from 0 to _n_in_ - 1
    - second-level _F*_ attribute indicates a nonlinear block; this key should match the block position in _type_. There are zero or more such blocks, but currently two.
        - third_level _activation_ attribute may be _tanh_, _sigmoid_ or _relu_
        - third_level _n_in_ attribute is the number of input channels
        - third_level _n_hid_ attribute is the number of neurons in the hidden layer
        - third_level _n_out_ attribute is the number of output channels
        - third_level _net.0.weight_ attribute is a 2D array of input layer weight coefficients, one row for each hidden-layer neuron, one column for each input channel
        - third_level _net.0.bias_ attribute is a 1D array of input layer bias coefficients, one for each hidden-layer neuron
        - third_level _net.2.weight_ attribute is a 2D array of output layer weight coefficients, one row for each output channel, one column for each hidden-layer neuron
        - third_level _net.2.bias_ attribute is a 1D array of output layer bias coefficients, one for each output channel

### HELICS Example

HELICS 3.0 and the Python wrapper must be installed:

    pip install helics

To run the case: 

    python pv1_configure_helics.py
    pv1_helics

To plot the results shown below:

    python pv1_plot.py

The relevant files are:

- _*.log_ contains text and debugging output from the HELICS federates
- _clean.bat_ deletes the log and json files mentioned in this list
- _helics_player.txt_ read by the _helics_player_, contains weather and control inputs for the simulation
- _kill23404.bat_ helper script to halt a HELICS federation that didn't exit cleanly. Assumes each federate and the broker use port 23404, which is the default. Call this once for each federate that's still running. If a federate refuses to exit (on Windows) just wait about 30 seconds.
- _list23404.bat_ helper script to list all HELICS federates that are listening to port 23404.
- _pv1_client.json_ contains HELICS publication, subscription, and other configurations for _pv1_client.py_
- _pv1_client.py_ a HELICS federate that constructs _Vrms_ from _Irms_ and _Rgrid_, and collects the simulation data from _pv1_server.py_. It takes the role of an EMT simulator using the HW-PV model.
- _pv1_configure_helics.py_ produces _pv1_client.json_, _pv1_server.json_, and _pv1_helics.bat_. Reads _pv1_fhf_poly.json_ for model information.
- _pv1_helics.bat_ launches the HELICS broker and three federates (player, client, and server)
- _pv1_plot.py_ makes a quick plot from _pv1_server.hdf5_
- _pv1_server.hdf5_ output data from _pv1_server.py_ in a Pandas DataFrame
- _pv1_server.json_ contains HELICS publication, subscription, and other configurations for _pv1_server.py_
- _pv1_server.py_ is a HELICS federate that subscribes to all weather and control inputs, runs the HW-PV model, and publishes the HW-PV model outputs. In this example, the input and output data are saved from this federate for convenience.

![HELICS_Fed](/examples/hwpv/HELICS_Federation.png)

**HELICS Federates and Message Topics**

![HELICS_Sim](/examples/hwpv/pv1_helics.png)

**System Simulation using HELICS**

### Other Code Files

These Python files are used to train and validate HW models, but the sample data to use them is not located in this repository:

- _common.py_ implements a batch-oriented dataset loader for training.
- _pv1_export.py_ writes the model coefficients and scaling factors to a single JSON file
- _pv1_metrics.py_ writes the RMS errors for each output channel, by case and also for the total
- _pv1_test.py_ plots one or more training datasets, comparing true and estimated outputs
- _pv1_test_iir.py_ plots one or more training datasets, comparing true and estimated outputs, overlaying the IIR simulation
- _pv1_training.py_ trains the HW model, determines channel scaling, saves the model to PKL files and a JSON file of scaling factors

## License

See [License](license.txt)

## Notice

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

    PACIFIC NORTHWEST NATIONAL LABORATORY
                operated by
                 BATTELLE
                 for the
     UNITED STATES DEPARTMENT OF ENERGY
      under Contract DE-AC05-76RL01830

Copyright 2021-2022, Battelle Memorial Institute