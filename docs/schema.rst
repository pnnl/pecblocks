.. _schema:

Schema
======

These are JSON schemas for the model configuration, normalization factors, 
sensitivity loss configuration, clamping loss configuration, and trained 
model blocks. These appear in various files for a specific *model*:

* In **model_config.json**, the section `HWPV Model Configuration`_ must appear. Optionally, `Sensitivity Losses`_ and `Clamping Losses`_ may appear. **The user must create this file.**
* A training run creates **normfacs.json** at the beginning. The section `Normalization Factors`_ appears. The *pecblocks* package creates this file in a subdirectory that was specified in *model_config.json*.
* A model export creates **model_fhf.json** in a subdirectory that was specified in *model_config.json*.  The sections `Block F1 or F2`_, `Block H1`_, `Block H1s`_, and `Block Q1s`_ appear. In addition, recognized sections from *model_config.json* and *normfacs.json* will be copied into *model_fhf.json*. The *pecblocks* package creates this file in the same directory as `normfacs.json`.

The user may add other JSON objects to *model_config.json*, but these are 
not copied into *model_folder/model_fhf.json* during model export. 

.. jsonschema:: config.json

.. jsonschema:: normalization.json

.. jsonschema:: sensitivity.json

.. jsonschema:: clamping.json

.. jsonschema:: blockF1.json

.. jsonschema:: blockH1.json

.. jsonschema:: blockH1s.json

.. jsonschema:: blockQ1s.json


