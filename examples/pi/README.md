## HWPV Lightweight Model Evaluation

Evaluates a trained HWPV model using only numpy, suitable
for implementation on Raspberry Pi.

The scripts and test files currently used in this example are:

- _hwpv\_evaluator.py_ reads the HPWV model from a JSON file, and evaluates it at a time step.
- _hwpv\_pi.py_ test harness for _hwpv\_evaluator.py_, creates _hwpv\_pi.hdf5_
- _pi\_plot.py_ plots the data from _hwpv\_pi.hdf5_.
- _../hwpv/big/balanced\_fhf.json_, from this repository, contains the model parameters for testing.

To run this example:

    python hwpv_pi.py
    python pi_plot.py

### License

See [License](../../license.txt)

### Notice

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

    PACIFIC NORTHWEST NATIONAL LABORATORY
                operated by
                 BATTELLE
                 for the
     UNITED STATES DEPARTMENT OF ENERGY
      under Contract DE-AC05-76RL01830

Copyright 2021-2023, Battelle Memorial Institute