import json
import sys
import os

if __name__ == '__main__':
  Tmax = 8.0
  fname_in = '../hwpv/models/pv1_fhf_poly.json'
  fname_out1 = 'pv1_server.json'
  fname_out2 = 'pv1_client.json'
  fname_bat = 'pv1_helics.bat'
  fname_sh = 'pv1_helics.sh'
  if len(sys.argv) > 1:
    fname_in = sys.argv[1]
  fp = open (fname_in, 'r')
  cfg_app = json.load (fp)
  fp.close()

  cfg_app['Tmax'] = Tmax
  cfg_server = {}
  cfg_server['name'] = 'pv1_server'
  cfg_server['core_type'] = 'zmq'
  cfg_server['period'] = cfg_app['t_step']
  cfg_server['log_level'] = 'none'
  pubs = []
  pubs.append({'global':False,'key':'vdc', 'type':'double'})
  pubs.append({'global':False,'key':'idc', 'type':'double'})
  pubs.append({'global':False,'key':'Vs', 'type':'complex'})
  pubs.append({'global':False,'key':'Is', 'type':'complex'})
  pubs.append({'global':False,'key':'Ic', 'type':'complex'})
  cfg_server['publications'] = pubs
  subs = []
  subs.append({'key':'player/G', 'type':'double', 'required':True})
  subs.append({'key':'player/T', 'type':'double', 'required':True})
  subs.append({'key':'player/Ud', 'type':'double', 'required':True})
  subs.append({'key':'player/Fc', 'type':'double', 'required':True})
  subs.append({'key':'pv1_client/Vrms', 'type':'complex', 'required':True})
  subs.append({'key':'player/ctl', 'type':'double', 'required':True})
  cfg_server['subscriptions'] = subs
  cfg_server['application'] = cfg_app
  fp = open (fname_out1, 'w')
  json.dump (cfg_server, fp, indent=2)
  fp.close()

  cfg_client = {}
  cfg_client['name'] = 'pv1_client'
  cfg_client['core_type'] = 'zmq'
  cfg_client['period'] = cfg_app['t_step']
  cfg_client['log_level'] = 'none'
  pubs = []
  cfg_client['publications'] = pubs
  pubs.append({'global':False,'key':'Vrms', 'type':'complex'})
  subs = []
  cfg_client['subscriptions'] = subs
  subs.append({'key':'pv1_server/Vs', 'type':'complex', 'required':True})
  subs.append({'key':'pv1_server/Is', 'type':'complex', 'required':True})
  subs.append({'key':'pv1_server/Ic', 'type':'complex', 'required':True})
  subs.append({'key':'pv1_server/vdc', 'type':'double', 'required':True})
  subs.append({'key':'pv1_server/idc', 'type':'double', 'required':True})
  subs.append({'key':'player/Rg', 'type':'double', 'required':True})
  cfg_client['application'] = {'Tmax':Tmax}
  fp = open (fname_out2, 'w')
  json.dump (cfg_client, fp, indent=2)
  fp.close()

  fp = open (fname_bat, 'w')
  fp.write('start /b cmd /c helics_broker -f 3 --name=mainbroker ^>broker.log 2^>^&1\n')
  fp.write('start /b cmd /c helics_player -n player --input=helics_player.txt --local --time_units=s --stop {:.3f}s ^>player.log 2^>^&1\n'.format(Tmax))
  fp.write('start /b cmd /c python pv1_client.py ^>client.log 2^>^&1\n')
  fp.write('start /b cmd /c python pv1_server.py ^>server.log 2^>^&1\n')
  fp.close()

  fp = open (fname_sh, 'w')
  fp.write('(exec helics_broker -f 3 --name=mainbroker &> broker.log &)\n')
  fp.write('(exec helics_player -n player --input=helics_player.txt --local --time_units=s --stop {:.3f}s &> player.log &)\n'.format(Tmax))
  fp.write('(exec python3 pv1_client.py &> client.log &)\n')
  fp.write('(exec python3 pv1_server.py &> server.log &)\n')
  fp.close()

  os.chmod (fname_sh, 0o755)


