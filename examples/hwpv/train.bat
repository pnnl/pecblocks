if not exist %1\ mkdir %1

python pv3_training.py %1_config.json
python loss_plot.py %1 -1
