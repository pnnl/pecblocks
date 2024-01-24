#!/bin/bash

(exec helics_broker -f 2 --name=mainbroker --loglevel debug &> broker.log &)
(exec helics_player helics_player.txt -n player --local --time_units=s --stop 8.000s --loglevel debug &> player.log &)
(exec helics_recorder --config-file helics_recorder.txt --stop 8.000s --loglevel debug &> recorder.log &)

