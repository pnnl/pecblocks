# Copyright (C) 2024 Battelle Memorial Institute

import sys

if __name__ == '__main__':
  dt = 0.01
  ics = {'G': 0.0, 'T': 35.0, 'Fc': 60.0, 'Md': 1.0, 'Mq': 0.0, 'Ctl': 0.0, 'Ra': 90.0, 'Rb': 90.0, 'Rc': 90.0}
  steps = {'Ctl':[1.0, 1.0], 'Ra':[65.0, 6.0], 'Rb':[65.0, 6.0], 'Rc':[65.0, 6.0]}
  Gramp = {'t_start':0.1, 't_end': 1.1, 'Gfinal': 1000.0}
  caps = {'Ctl':[1.0, 8.0]}

  fname = 'ucf1.txt'
  if len(sys.argv) > 1:
    fname = sys.argv[1]

  fp = open (fname, 'w')

  print ('#second topic type(d,s,i,c) value', file=fp)
  for key, val in ics.items():
    print ('-1.00 {:3s} d {:10.4f}'.format (key, val), file=fp)

  t = Gramp['t_start']
  G = ics['G']
  dG = dt * (Gramp['Gfinal'] - G) / (Gramp['t_end'] - t)
  while G <= Gramp['Gfinal']:
    print ('{:.3f} G   d {:10.4f}'.format (t, G), file=fp)
    G += dG
    t += dt

  for key, val in steps.items():
    v1 = ics[key]
    v2 = val[0]
    t = val[1]
    print ('{:.3f} {:3s} d {:10.4f}'.format (t-dt, key, v1), file=fp)
    print ('{:.3f} {:3s} d {:10.4f}'.format (t, key, v2), file=fp)

  for key, val in caps.items():
    v = val[0]
    t = val[1]
    print ('{:.3f} {:3s} d {:10.4f}'.format (t-dt, key, v), file=fp)
    print ('{:.3f} {:3s} d {:10.4f}'.format (t, key, v), file=fp)

  fp.close()


