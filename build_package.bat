rem copy selected Python files to the end-user example directory, build a lightweight package
cd docs
call make clean
cd ..

xcopy /d /y examples\hwpv\loss_plot.py    examples\training
xcopy /d /y examples\hwpv\pv3_export.py   examples\training
xcopy /d /y examples\hwpv\pv3_metrics.py  examples\training
xcopy /d /y examples\hwpv\pv3_sigma.py    examples\training
xcopy /d /y examples\hwpv\pv3_test.py     examples\training
xcopy /d /y examples\hwpv\pv3_training.py examples\training
xcopy /d /y examples\hwpv\train.bat       examples\training
xcopy /d /y examples\hwpv\train.sh        examples\training
xcopy /d /y examples\hwpv\export.bat      examples\training
xcopy /d /y examples\hwpv\export.sh       examples\training

rd /s /q dist
python -m build
twine check dist/*

