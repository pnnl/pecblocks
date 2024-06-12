"""
  Power Electronic Converter generalized block diagram models (pecblocks).

  These models are based on a Hammerstein-Wiener framework for photovoltaics (HWPV).
  The training data may come from electromagnetic transient (EMT) simulations 
  or laboratory tests of PV inverters, either grid-forming (GFM) or grid-following (GFL).
  The inverters may be three-phase or single-phase.

  The required packages include *dynonet*, *harold*, *control*, and *h5py*.
  These in turn require *torch*, *pandas*, *scipy*, *numpy*, and *matplotlib*.

  Contains these implemented modules:

    - **common.py**: supports a PyTorch dataloader customized to fitting generalized block diagram models.  Implements **PVInvDataset** class.
    - **pv3_functions.py**: analyzes the closed-loop stability of trained Norton and Thevenin HWPV models.
    - **pv3_poly.py**: the main class for training, exporting, and evaluating HWPV models.  Implements **pv3_poly** class.
    - **util.py**: reads training data from HDF5 and CSV files into Pandas DataFrames.

  See https://pecblocks.readthedocs.io/en/latest/ for documentation.
"""

from .version import __version__
