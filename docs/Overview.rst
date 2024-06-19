.. role:: math(raw)
   :format: html latex
..

Overview
========

Power Electronic Converter Blocks (pecblocks) use the output of detailed 
electromagnetic transient (EMT) simulations to produce generalized block 
diagram models of power electronic systems. The process uses deep learning 
with customized block architectures. The outputs are reduced-order models 
that meet specified accuracy requirements, while providing important 
advantages over the original EMT models: 

* Converter Block models have fewer nodes and can take longer time steps, resulting in shorter simulation times
* Converter Block models are continuously differentiable, making them compatible with control design methods

Models will run in both time domain and frequency domain. The scope 
includes not only the power electronic converter, but also the 
photovoltaic (PV) array, maximum power point tracking (MPPT), phase-lock 
loop (PLL) control, output filter circuits, battery storage if present, 
etc. The applications include but may not be limited to solar power 
inverters, energy storage converters, motor drives, and other power 
electronics equipment. 

Background
----------

For technical background on *pecblocks*, see `SysDO 2024 Paper (submitted) <_static/paper.pdf>`_

For technical background on *dynoNet*, see `Forgione, Piga Paper <https://arxiv.org/pdf/2006.02250>`_

For technical background on grid-forming inverters, see `Rathnayake, et. al. <https://doi.org/10.1109/ACCESS.2021.3104617>`_

Installation
------------

To install the Python package::

    pip install pecblocks

Quick Start
-----------

The package includes two examples. From the *example/training* directory::

    download.bat or ./download.sh
    train ucf3 or ./train.sh ucf3
    export ucf3 or ./export.sh ucf3
    python pv3_test.py ucf3_config.json

The first command will download a 90-MB sample data file. The second command trains
a model, which may take several minutes. The third command exports the trained model
for s-domain simulations. Results appear in the *examples/training/ucf3* directory.
The fourth command plots a sample comparison of estimated vs. true output, for
one of the data cases.

From the *example/sdomain* directory::

    go.bat or ./go.sh

This runs a continous-time simulation of a trained HWPV model, at a longer time
step than the original z-domain model was trained at.

Example Repository
------------------

See `GitHub Examples directory <https://github.com/pnnl/pecblocks/tree/master/examples>`_


