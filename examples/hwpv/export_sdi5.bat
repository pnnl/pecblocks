python pv3_export.py ./sdi5/sdi5_config.json > sdi5\metrics.txt
python pv3_metrics.py ./sdi5/sdi5_config.json c:/data/sdi5.hdf5 >> sdi5\metrics.txt
rem python pv3_export.py ./sdi5a/sdi5a_config.json > sdi5a\metrics.txt
rem python pv3_metrics.py ./sdi5a/sdi5a_config.json c:/data/sdi5a.hdf5 >> sdi5a\metrics.txt
rem python pv3_export.py ./sdi5b/sdi5b_config.json > sdi5b\metrics.txt
rem python pv3_metrics.py ./sdi5b/sdi5b_config.json c:/data/sdi5b.hdf5 >> sdi5b\metrics.txt

