rem call export.bat ucfB_n_2nd_Ctrl
call export.bat ucfB_t_2nd_Ctrl
call export.bat ucfB_t_2nd_Step
call export.bat ucfB_t_2nd_Ramp
rem call export.bat sdi5
rem call export.bat osg4
rem call export.bat ucf2
rem call export.bat unb3
rem call export.bat bal3
rem python pv3_lambda.py ucfB_n_2nd_Ctrl_config.json
python pv3_lambda.py ucfB_t_2nd_Ctrl_config.json
python pv3_lambda.py ucfB_t_2nd_Step_config.json
python pv3_lambda.py ucfB_t_2nd_Ramp_config.json



