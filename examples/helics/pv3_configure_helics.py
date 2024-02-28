# Copyright (C) 2022-2024 Battelle Memorial Institute
# configure a three-phase HELICS simulation
#  arg1: the player[.txt] file of inputs
#  arg2: the model coefficient JSON file

import json
import sys
import os

if __name__ == '__main__':
  Tmax = 8.0
  fname_in = '../simscape/balanced_fhf.json'
  fname_in = 'big/balanced_fhf.json'
  fname_in = '../hwpv/ucf3/ucf3_fhf.json'
  fname_out1 = 'pv3_server.json'
  fname_out2 = 'pv3_client.json'
  fname_bat = 'pv3_helics.bat'
  fname_sh = 'pv3_helics.sh'
  case = 'ucf1' # 'flat3' # 'ramp3'
  if len(sys.argv) > 1:
    case = sys.argv[1]
    if len(sys.argv) > 2:
      fname_in = sys.argv[2]
  fp = open (fname_in, 'r')
  cfg_app = json.load (fp)
  fp.close()

  cfg_app['Tmax'] = Tmax
  cfg_server = {}
  cfg_server['name'] = 'pv3_server'
  cfg_server['core_type'] = 'zmq'
  cfg_server['period'] = cfg_app['t_step']
  cfg_server['log_level'] = 'none'
  pubs = []
  pubs.append({'global':False,'key':'Vdc', 'type':'double'})
  pubs.append({'global':False,'key':'Idc', 'type':'double'})
  pubs.append({'global':False,'key':'Id', 'type':'double'})
  pubs.append({'global':False,'key':'Iq', 'type':'double'})
  cfg_server['publications'] = pubs
  subs = []
  subs.append({'key':'player/G', 'type':'double', 'required':True})
  subs.append({'key':'player/T', 'type':'double', 'required':True})
  subs.append({'key':'player/Md', 'type':'double', 'required':True})
  subs.append({'key':'player/Mq', 'type':'double', 'required':True})
  subs.append({'key':'player/Fc', 'type':'double', 'required':True})
  subs.append({'key':'player/Ctl', 'type':'double', 'required':True})
  subs.append({'key':'pv3_client/Vd', 'type':'double', 'required':True})
  subs.append({'key':'pv3_client/Vq', 'type':'double', 'required':True})
  cfg_server['subscriptions'] = subs
  cfg_server['application'] = cfg_app
  fp = open (fname_out1, 'w')
  json.dump (cfg_server, fp, indent=2)
  fp.close()

  cfg_client = {}
  cfg_client['name'] = 'pv3_client'
  cfg_client['core_type'] = 'zmq'
  cfg_client['period'] = cfg_app['t_step']
  cfg_client['log_level'] = 'none'
  pubs = []
  cfg_client['publications'] = pubs
  pubs.append({'global':False,'key':'Vd', 'type':'double'})
  pubs.append({'global':False,'key':'Vq', 'type':'double'})
  subs = []
  cfg_client['subscriptions'] = subs
  subs.append({'key':'pv3_server/Id', 'type':'double', 'required':True})
  subs.append({'key':'pv3_server/Iq', 'type':'double', 'required':True})
  subs.append({'key':'pv3_server/Vdc', 'type':'double', 'required':True})
  subs.append({'key':'pv3_server/Idc', 'type':'double', 'required':True})
  subs.append({'key':'player/Ra', 'type':'double', 'required':True})
  subs.append({'key':'player/Rb', 'type':'double', 'required':True})
  subs.append({'key':'player/Rc', 'type':'double', 'required':True})
  cfg_client['application'] = {'Tmax':Tmax}
  fp = open (fname_out2, 'w')
  json.dump (cfg_client, fp, indent=2)
  fp.close()

  fp = open (fname_bat, 'w')
  fp.write('start /b cmd /c helics_broker -f 3 --name=mainbroker ^>broker.log 2^>^&1\n')
  fp.write('start /b cmd /c helics_player -n player --input={:s}.txt --local --time_units=s --stop {:.3f}s ^>player.log 2^>^&1\n'.format(case, Tmax))
  fp.write('start /b cmd /c python pv3_client.py ^>client.log 2^>^&1\n')
  fp.write('start /b cmd /c python pv3_server.py ^>server.log 2^>^&1\n')
  fp.close()

  fp = open (fname_sh, 'w')
  fp.write('(exec helics_broker -f 3 --name=mainbroker &> broker.log &)\n')
  fp.write('(exec helics_player -n player --input={:s}.txt --local --time_units=s --stop {:.3f}s &> player.log &)\n'.format(case, Tmax))
  fp.write('(exec python3 pv3_client.py &> client.log &)\n')
  fp.write('(exec python3 pv3_server.py &> server.log &)\n')
  fp.close()

  os.chmod (fname_sh, 0o755)


