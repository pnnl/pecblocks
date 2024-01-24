#### HELICS Player Test

This example tests HELICS player and recorder functionality. It does not require Python, pecblocks, nor the Python helics package. Instead,
install the HELICS binary applications from the [Release Page](https://github.com/GMLC-TDC/HELICS/releases)

To run the test on Windows: 

    test_player

To run the test on Linux or Mac:

    ./test_player.sh

The expected output is a file _out.txt_ with captured HELICS values:

- there should be a ramp in _player/G_ from 0 to 950, between 0 and 0.2 s
- there should be a step in _player/ctl_ from 0 to 1, at 2.0 s
- there should be a ramp in _player/G_ from 950 to 825, between 2.40 and 2.425 s
- there should be a step in _player/T_ from 5 to 35, at 2.45 s
- there should be a step in _player/Fc_ from 60 to 63, at 4.45 s
- there should be a step in _player/Ud_ from 1 to 0.92, at 5.45 s
- there should be a step in _player/Rg_ from 4.25 to 6.25, at 6.45 s

However, in HELICS v3.4 and some earlier versions, the tags are always _player/Fc_.

The relevant files are:

- _\*.log_ contains text and debugging output from the HELICS federates
- _clean.bat/sh_ deletes the log and output files
- _helics_player.txt_ read by the _helics_player_, contains weather and control inputs for the simulation
- _helics_recorder.txt_ read by the _helics_recorder_, configures the data to capture
- _kill23404.bat_ Windows helper script to halt a HELICS federation that didn't exit cleanly. Assumes each federate and the broker use port 23404, which is the default. Call this once for each federate that's still running. If a federate refuses to exit (on Windows) just wait about 30 seconds.
- _list23404.bat_ Windows helper script to list all HELICS federates that are listening to port 23404.
- _out.txt_ contains the captured HELICS publications
- _test_player.bat/sh_ launches the HELICS broker and two federates (player and recorder)

