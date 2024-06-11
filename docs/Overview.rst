.. role:: math(raw)
   :format: html latex
..

Overview
========

Power Electronic Converter Blocks (pecblocks) use the output of detailed electromagnetic transient (EMT) simulations to produce generalized block diagram models of power electronic systems. The process uses deep learning with customized block architectures. The outputs are reduced-order models that meet specified accuracy requirements, while providing important advantages over the original EMT models:

* Converter Block models have fewer nodes and can take longer time steps, resulting in shorter simulation times
* Converter Block models are continuously differentiable, making them compatible with control design methods

Models will run in both time domain and frequency domain. The scope includes not only the power electronic converter, but also the photovoltaic (PV) array, maximum power point tracking (MPPT), phase-lock loop (PLL) control, output filter circuits, battery storage if present, etc. The applications include but may not be limited to solar power inverters, energy storage converters, motor drives, and other power electronics equipment.

-----
Paper
-----

For technical background, see `SysDO 2024 Paper (submitted) <_static/paper.pdf>`_

------------
Installation
------------

To install the Python package::

    pip install pecblocks

--------
Examples
--------

See `GitHub Examples directory <https://github.com/pnnl/pecblocks/tree/master/examples>`_


