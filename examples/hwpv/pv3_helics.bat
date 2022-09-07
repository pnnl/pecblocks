start /b cmd /c helics_broker -f 3 --name=mainbroker ^>broker.log 2^>^&1
start /b cmd /c helics_player --input=flat3.txt --local --time_units=s --stop 8.000s^>player.log 2^>^&1
start /b cmd /c python pv3_client.py ^>client.log 2^>^&1
start /b cmd /c python pv3_server.py ^>server.log 2^>^&1
