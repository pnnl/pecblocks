# Examples from Continuation Report 

These examples are based on outputs from the Alternative Transients Program (ATP), archived in the ```./data``` subdirectory. 

Steps to run:

1. ```python plot_training_data.py``` displays plots of the training data used in the next two steps. PDF plots are saved in the ```./models/training``` subdirectory.
2. ```python training1.py``` fits a simple step response. PDF plots and model PKL files are saved in the ```./models/test``` subdirectory.
3. ```python test_hw.py``` repeats 7 examples used in the continuation report. PDF plots and model PKL files are saved in subdirectories like ```./models/TtoIdc```.

For customization, see ```test_hw.py```:

1. Block fitting parameters are set in the ```training_sets``` dictionary, beginning around line 18.
2. The model architecture is a series cascade of nonlinear, linear, nonlinear blocks (FGF). To change this, edit the ```def model``` function definition that begins around line 186. See the dynoNet examples and API documentation for more guidance.
