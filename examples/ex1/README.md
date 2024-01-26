### Examples from Project Continuation Report 

These examples are based on outputs from the Alternative Transients Program (ATP), archived in the ```./data``` subdirectory. 

Steps to run:

1. ```python plot_training_data.py``` displays plots of the training data used in the next two steps. PDF plots are saved in the ```./models/training``` subdirectory.
2. ```python training1.py``` fits a simple step response. PDF plots and model PKL files are saved in the ```./models/test``` subdirectory.
3. ```python test_hw.py``` repeats 7 examples used in the continuation report. PDF plots and model PKL files are saved in subdirectories like ```./models/TtoIdc```.

For customization, see ```test_hw.py```:

1. Block fitting parameters are set in the ```training_sets``` dictionary, beginning around line 18.
2. The model architecture is a series cascade of nonlinear, linear, nonlinear blocks (FGF). To change this, edit the ```def model``` function definition that begins around line 186. See the dynoNet examples and API documentation for more guidance.

To save the learned attributes in ```models.json```:

1. ```python export_models.py```

To plot impulse responses of saved models:

1. ```python G1_impulse.py```
2. Plots ```GtoVdc```, but a different model can be specified at Line 25.

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

Copyright 2021-2024, Battelle Memorial Institute
