start /b cmd /c helics_broker -f 2 --name=mainbroker --loglevel trace ^>broker.log 2^>^&1
start /b cmd /c helics_player helics_player.txt -n player --local --stop 8s --loglevel debug ^>player.log 2^>^&1
start /b cmd /c helics_recorder helics_recorder.txt --verbose --stop 8s --loglevel debug ^>recorder.log 2^>^&1

