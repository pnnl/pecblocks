# pecblocks

Power Electronic Converter Blocks (pecblocks) use the output of detailed electromagnetic transient (EMT) simulations to produce generalized block
diagram models of power electronic systems. The process uses deep learning with customized block architectures. The
outputs are reduced-order models that meet specified accuracy requirements, while providing important advantages over
the original EMT models:

- ConverterBlock models have fewer nodes and can take longer time steps, resulting in shorter simulation times
- ConverterBlock models are continuously differentiable, making them compatible with control design methods

Models will run in both time domain and frequency domain. The scope includes not only the power electronic converter,
but also the photovoltaic (PV) array, maximum power point tracking (MPPT), phase-lock loop (PLL) control, output filter
circuits, battery storage if present, etc. The applications include but may not be limited to solar power inverters, energy
storage converters, motor drives, and other power electronics equipment.

## Panels-to-Grid MIMO Test Case

In the directory _examples/hwpv_, a multiple-input, multiple-output HW model has been
fitted to the PV panels, MPPT, DC/DC, inverter PWM switching, LCL output filter, and RMS measurements.
It was based on ATP simulations with five inputs, _G_, _T_, _Ud_, _Fc_, and _Vrms_. A sixth input is
created as a polynomial feature, defined as _GVrms_. The outputs are _Idc_, _Vdc_, and _Irms_.

The Python files currently used are:

- _pv1_poly.py_ implements a multi-channel Hammerstein-Wiener architecture. Earlier architectures are in the _archive_ subdirectory.
- _common.py_ implements a batch-oriented dataset loader for training.
- _pv1_training.py_ trains the HW model, determines channel scaling, saves the model to PKL files and a JSON file of scaling factors
- _pv1_export.py_ writes the model coefficients and scaling factors to a single JSON file
- _pv1_test.py_ plots one or more training datasets, comparing true and estimated outputs
- _pv1_metrics.py_ writes the RMS errors for each output channel, by case and also for the total

The input files are:

- _data/pv1.hdf5_ 72 training cases from ATP simulation, saved at 0.2-ms time steps. The _pecblocks_ module helps to read this file.
- _models/pv1_config.json_ contains some adjustable parameters for the HW model and training:
    - _COL_T_ is the column name for time in the _hdf5_ file. Unlikely to change.
    - _COL_U_ are the column names for ATP inputs in the _hdf5_ file. Unlikely to change. Both _G_ and _Vrms_ should exist, because they are used to create a polynomial feature.
    - _COL_Y_ are the column names for ATP outputs in the _hdf5_ file. Unlikely to change.
    - _lr_ is the learning rate
    - _num_iter_ is the number of training iterations
    - _print_freq_ determines how often the iteration loss is written to console
    - _batch_size_ for training should divide equally into the number of training cases
    - _n_skip_ is the number of time steps to skip at the beginning of each training case. Should be at least 1.
    - _n_trunc_ is the number of time steps to truncate from the end of each training case.
    - _n_dec_ is the decimation number, i.e., a downsampling of the ATP data, which defines the HW model time step
    - _na_ is the order of denominator polynomials in H1(z)
    - _nb_ is the order of numerator polynomials in H1(z)
    - _nk_ is the number of discrete step delays in H1(z)
    - _activation_ is the activation function in F1 and F2. _relu_ converges faster, but _tanh_ (best) and _sigmoid_ are smoother.
    - _nh1_ is the number of hidden-layer neurons in F1
    - _nh2_ is the number of hidden-layer neurons in F2

The output files from training are:
    - _models/normfacs.json_ contains a _scale_ and _offset_ for each _COL_U_ and _COL_Y_. Reading from _hdf5_, subtract _offset_ from the values and then divide by _scale_. The HW model is trained to these normalized values. Writing from normalized values to physical quantities, multiply by _scale_ and then add _offset_.
    - _models/F1.pkl_, _models/H1.pkl_, and _models/F2.pkl_ contain the trained model coefficients in a binary format.
    - _models/FHF_train_loss.pdf_ is a graph of the training loss by iteration

After training, run _python pv1_export.py_ to create _models/pv1_fhf_poly.json_, which contains the following in a readable text format.

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