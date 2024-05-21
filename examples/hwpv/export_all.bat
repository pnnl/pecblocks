call export.bat ucf4siir
call export.bat ucf4s2nd
call export.bat ucf4tsiir
call export.bat ucf4ts2nd
rem call export.bat sdi5
rem call export.bat osg4
rem call export.bat ucf2
rem call export.bat unb3
rem call export.bat bal3
python pv3_lambda.py ucf4siir_config.json
python pv3_lambda.py ucf4s2nd_config.json
python pv3_lambda.py ucf4tsiir_config.json
python pv3_lambda.py ucf4ts2nd_config.json



