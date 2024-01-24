#!/bin/bash

(exec helics_broker -f 2 --name=mainbroker --loglevel trace &> broker.log &)
(exec helics_player helics_player.txt -n player --local --stop 8s --loglevel debug &> player.log &)
(exec helics_recorder helics_recorder.txt --verbose --stop 8s --loglevel debug &> recorder.log &)

