start /b cmd /c helics_broker -f 2 --name=mainbroker --loglevel summary ^>broker.log 2^>^&1
start /b cmd /c helics_player helics_player.txt -n player --local --time_units=s --stop 8s --loglevel connections ^>player.log 2^>^&1
start /b cmd /c helics_recorder --config-file helics_recorder.txt --stop 8s --loglevel connections --verbose ^>recorder.log 2^>^&1

