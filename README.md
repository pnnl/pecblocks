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

## User Instructions

See [ReadTheDocs Manual](https://pecblocks.readthedocs.io/en/latest/)

## Developer Instructions

Familiarity with Python, `git` and `sphinx` is expected. The developer may need credentials for this
project on GitHub, ReadTheDocs, and/or PyPi. The application program interface (API) and the schema
for JSON files are documented in the [ReadTheDocs Manual](https://pecblocks.readthedocs.io/en/latest/).

Python 3.7.6 and later have been used for testing. From a command prompt in this directory,
Install the necessary Python modules with:

- `git clone https://github.com/pnnl/pecblocks.git`
- `cd pecblocks`
- `pip install -r requirements.txt`
- `pip install -e .`

The project on ReadTheDocs will re-build automatically upon commits to the git repository. To build and run the documentation locally:

- Change to the `docs` subdirectory of your git clone 
- The first time only, invoke `pip install -r requirements.txt --upgrade-strategy only-if-needed`
- `make html`
- From a browser, open the file `docs\_build\html\index.html` from the directory of your git clone

To deploy the project on PyPi, staring in the directory of your git clone, where `setup.py` is located:

- Make sure that the version number in `setup.cfg` and `src\pecblocks\version.py` is new.
- Invoke `rd /s /q dist` on Windows, or `rm -rf dist` on Linux or Mac OS X
- Invoke `pip install build` and `pip install twine` if necessary
- `python -m build`
- `twine check dist/*` should not show any errors
- `twine upload -r testpypi dist/*` requires project credentials for pecblocks on test.pypi.org (Note: this will reject if version already exists, also note that testpypi is a separate register to pypi)
- `pip install -i https://test.pypi.org/simple/ pecblocks==0.0.3` for local testing of the deployable package, example version 0.0.3 (Note: consider doing this in a separate Python test environment)
- `twine upload dist/*` for final deployment; requires project credentials for pecblocks on pypi.org. If 2-Factor-Authentication is enabled an [API token](https://pypi.org/help/#apitoken>) needs to be used.

## Directories

- _data_ contains the training data used in project publications and examples
- _docs_ contains the source files for user documentation
- _examples_ contains the end-user scripts and configuration files to run various examples
- _src_ contains the Python package code to be deployed on PyPi

In the _examples_ subdirectory:

- _data\_prep_ processes ATP, lab, and newer Simscape outputs into a format form pecblocks
- _dev_ contains archived examples, scripts, and models for developing the deployable package, and investigating Thevenin vs. Norton model sensitivity (_no longer actively used_)
- _dll_ contains C/C++ and Python code for standalone evaluation of HWPV models in continuous time domain, using Backward Euler integration
- _ex1_ contains the first examples of inverter subsystem models (_no longer actively used_)
- _helics_ contains a co-simulation example with Python federates
- _hwpv_ contains the primary examples from project publications
- _initialization_ was used to develop better initialization of the linear _H1_ block (_no longer actively used_).
- _lcl_ uses phasor arithmetic to calculate voltages and currents at the inverter bridge terminals, from voltages and currents at the PCC. But this is an acausal operation, not good for control design. (_no longer actively used_)
- _media_ contains graphics of training data sets and output comparisons.
- _pi_ contains a lightweight model evaluation test for Raspberry Pi, using _numpy_ but not _torch_
- _pv1_ contains single-phase inverter modeling code, later incorporated into the generalized package for deployment (_no longer actively used_)
- _simscape_ processes MATLAB/Simscape outputs into a format for pecblocks (_no longer actively used_)

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

Copyright 2021-2024, Battelle Memorial Institute