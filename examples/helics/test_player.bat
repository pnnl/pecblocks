start /b cmd /c helics_broker -f 2 --name=mainbroker --loglevel data ^>broker.log 2^>^&1
start /b cmd /c helics_player helics_player.txt -n player --local --time_units=s --stop 8.000s --loglevel data ^>player.log 2^>^&1
rem start /b cmd /c helics_player --version ^>player.log 2^>^&1
start /b cmd /c helics_recorder --config-file helics_recorder.txt --stop 8.000s --loglevel data ^>recorder.log 2^>^&1

