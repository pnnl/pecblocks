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

To run this example:

    python pv1_import.py

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

### File Directory

The Python files currently used in this example are:

- _pv1_poly.py_ implements a multi-channel Hammerstein-Wiener architecture.
- _pv1_import.py_ reads the model for time-step simulation, and produces Bode plots

A sample trained model is provided in _pv1_fhf_poly.json_, which contains the following in a readable text format.

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
            - in the attribute name, _i_ is the input channel number, ranging from 0 to _n_in_ - 1
            - in the attribute name, _j_ is the output channel number, ranging from 0 to _n_out_ - 1
        - third-level _b_i_j_ attributes are arrays of numerator coefficients, of length equal to polynomial order, beginning with z-1. The implied z0 coefficient is always zero.
            - in the attribute name, _i_ is the input channel number, ranging from 0 to _n_in_ - 1
            - in the attribute name, _j_ is the output channel number, ranging from 0 to _n_out_ - 1
    - second-level _F*_ attribute indicates a nonlinear block; this key should match the block position in _type_. There are zero or more such blocks, but currently two.
        - third_level _activation_ attribute may be _tanh_, _sigmoid_ or _relu_
        - third_level _n_in_ attribute is the number of input channels
        - third_level _n_hid_ attribute is the number of neurons in the hidden layer
        - third_level _n_out_ attribute is the number of output channels
        - third_level _net.0.weight_ attribute is a 2D array of input layer weight coefficients, one row for each hidden-layer neuron, one column for each input channel
        - third_level _net.0.bias_ attribute is a 1D array of input layer bias coefficients, one for each hidden-layer neuron
        - third_level _net.2.weight_ attribute is a 2D array of output layer weight coefficients, one row for each output channel, one column for each hidden-layer neuron
        - third_level _net.2.bias_ attribute is a 1D array of output layer bias coefficients, one for each output channel

### Other Code Files

These Python files are used to train and validate HW models, but the sample data to use them is not located in this repository:

- _common.py_ implements a batch-oriented dataset loader for training.
- _pv1_training.py_ trains the HW model, determines channel scaling, saves the model to PKL files and a JSON file of scaling factors
- _pv1_export.py_ writes the model coefficients and scaling factors to a single JSON file
- _pv1_test.py_ plots one or more training datasets, comparing true and estimated outputs
- _pv1_metrics.py_ writes the RMS errors for each output channel, by case and also for the total

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