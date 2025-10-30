import os
import sys
import json
import numpy as np

TACS_template = """C control logic for HWPV models
C transfer functions, order N in columns 1-2, sign must be punched for all
C  non-blank inputs. If N=0 don't punch coefficients, it's a simultaneous sum.
C NAME=>  +IN1==> +IN2==> +IN3==> +IN4==> +IN5==> <=GAIN<=FXLO<=FXHI:NMLO>NMHI=>
C <=====N0<=======N1<=======N2<=======N3<=======N4<=======N5<=======N6<=======N7
C <=====D0<=======D1<=======D2<=======D3<=======D4<=======D5<=======D6<=======D7
C
C signal sources: FREQHZ, OMEGAR, ZERO, MINUS1, PLUS1, INFNTY, PI are built in
C                 TIMEX, DELTAT, ISTEP
C Type (col 1-2): 11=level, 14=cosine, 23=pulse, 24=ramp, 90=node V, 91=swt I
C                 92=sync mach, 93=swt status
C NAME=>  <=====AMPL<=====FREQ<==PHS DEG (not needed for 90-93)
C                   <===PERIOD<===PWIDTH (for 23 and 24)
C 
C supplemental devices (cols 1-2, specify group: 99 input, 88 inside, 98 output)
C NAME=>  = free-format FORTRAN expression after equal sign in column 11
C  operators are + - * / ** .OR. .AND. .NOT. .EQ. .NE. .LT. .LE. .GE. .GT.
C  functions are 
C     SIN COS TAN COTAN SINH COSH TANH (arguments in radians)
C     ASIN ACOS ATAN (answers in radians)
C     EXP LOG LOG10 SQRT ABS TRUNC MINUS INVRS RAD DEG 
C     SIGN (-1 if arg < 0, +1 otherwise)
C     NOT (0 if arg > 0, +1 otherwise)
C     SEQ6 (integer part of arg modulo 6)
C     RAN (returns a random number, argument is a seed or existing TACS name)
C
C NAME=>XX+IN1==> +IN2==> +IN3==> +IN4==> +IN5==> <====A<====B<====C:D===>E====>
C the group 88, 98, 99 goes in columns 1-2
C the code XX in columns 9-10 may be any of the following. 
C   50 frequency sensor
C   51 relay-operated switch
C   52 level-triggered switch
C   53 transport delay
C   54 pulse transport delay
C   55 digitizer
C   56 point-by-point nonlinearity (common)
C   57 multi-operation time-sequenced switch
C   58 controlled integrator (most common)
C   59 simple derivative
C   60 input-if component
C   61 signal selector
C   62 sample and track
C   63 instantaneous min/max
C   64 min/max tracking
C   65 accumulator and counter
C   66 rms value (beware, storage hog and freq must be specified)
C these are not used in HWPV; see the ATP Rule Book for more details
C
C TACS output requests, put a 33 in columns 1-2
C NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>NAME=>
C
C TACS initial conditions, put a 77 in columns 1-2
C NAME=>  <=====IVAL
C 
/TACS"""

Network_template = """C network interface for HWPV models
/BRANCH
C 345678901234567890123456789012345678901234567890123456789012345678901234567890
  __PCCA____IA              .001                                               0
  __PCCB____IB              .001                                               0
  __PCCC____IC              .001                                               0
  ____IA                    9.E4                                               0
  ____IB                    9.E4                                               0
  ____IC                    9.E4                                               0
/SOURCE
C < n 1><>< Ampl.  >< Freq.  ><Phase/T0><   A1   ><   T1   >< TSTART >< TSTOP  >
60____IA-1                                                               99999.0
60____IB-1                                                               99999.0
60____IC-1                                                               99999.0"""

DBM_start = """BEGIN NEW DATA CASE --NOSORT--
DATA BASE MODULE
$ERASE
ARG  __PCC, __OUT,_____T,_____G,____FC,____MD,____MQ
DUM ____WC,_THETA,_COSTH,_COSTM,_COSTP,_SINTH,_SINTM,_SINTP,____VD,____VQ
DUM ____V0,__VRMS,_GVRMS,_ARGDQ,__ARG0,____ID,____IQ,____I0,____IA,____IB
DUM ____IC"""

DBM_finish = """BEGIN NEW DATA CASE
C 
$PUNCH
BEGIN NEW DATA CASE
BLANK"""

tacs_names = set()

def make_tacs_name (s, bTrack=True):
  stacs = s.rjust(6,'_')
  if bTrack:
    tacs_names.add(stacs)
  return stacs

def AtpFit6(x):
  if x == 0.0:
    return '0.0000'
  elif x >= 10000:
    exp = 0
    while x >= 1000:
      x /= 10.0
      exp += 1
    xstr = '{:3d}.E{:d}'.format(int(round(x, 3)), exp)
  elif x >= 1000:
    xstr = '{:6.1f}'.format(x)
  elif x >= 100:
    xstr = '{:6.2f}'.format(x)
  elif x >= 10:
    xstr = '{:6.3f}'.format(x)
  elif x <= 0.001:
    exp = 0
    while x < 10.0:
      x *= 10.0
      exp += 1
    xstr = '{:2d}.E-{:d}'.format(int(round(x)), exp)
  else:
    xstr = '{:6.4f}'.format(x)
  return xstr

def make_tacs_array (coeffs):
#  return ''.join(['{:10.7f}'.format(x) for x in coeffs])
#  return ''.join(['{:s}'.format(AtpFit10(x)) for x in coeffs])
  return ''.join(['{:10.3e}'.format(x) for x in coeffs])

def make_tacs_fblock (F, tag, ltr, grp, fp):
  n_in = F['n_in']
  n_hid = F['n_hid']
  n_out = F['n_out']
  ltrA = ltr
  ltrB = str(chr(ord(ltr)+1))
  ltrC = str(chr(ord(ltr)+2))
  ltrD = str(chr(ord(ltr)+3))
  ltrE = str(chr(ord(ltr)+4))
  print ('C   input layer weights, net.0.weight', file=fp)
  vals = F['net.0.weight']
  for i in range(n_hid):
    for j in range(n_in):
      in_name = make_tacs_name('{:s}I{:d}'.format(tag, j))
      out_name = make_tacs_name('{:s}{:d}_{:d}'.format(ltrA, i,j))
      print ('{:s}{:s}  ={:12.9f}*{:s}'.format(grp, out_name, vals[i][j], in_name), file=fp)
  print ('C ====================================================', file=fp)
  print ('C   add up weighted layer inputs', file=fp)
  for i in range(n_hid):
    out_name = make_tacs_name('{:s}{:d}'.format(ltrB, i))
    print ('{:s}{:s}  ={:s}'.format(grp, out_name,
          '+'.join([make_tacs_name('{:s}{:d}_{:d}'.format(ltrA, i,j)) for j in range(n_in)])), file=fp)
  print ('C ====================================================', file=fp)
  print ('C   activation functions, apply net.0.bias', file=fp)
  vals = F['net.0.bias']
  for i in range(n_hid):
    out_name = make_tacs_name('{:s}{:d}'.format(ltrC, i))
    in_name = make_tacs_name('{:s}{:d}'.format(ltrB, i))
    print ('{:s}{:s}  = tanh({:12.9f}+{:s})'.format(grp, out_name, vals[i], in_name), file=fp)
  print ('C ====================================================', file=fp)
  print ('C   output layer weights, net.2.weight', file=fp)
  vals = F['net.2.weight']
  for i in range(n_out):
    for j in range(n_hid):
      in_name = make_tacs_name('{:s}{:d}'.format(ltrC, j))
      out_name = make_tacs_name('{:s}{:d}_{:d}'.format(ltrD, i,j))
      print ('{:s}{:s}  ={:12.9f}*{:s}'.format(grp, out_name, vals[i][j], in_name), file=fp)
  print ('C ====================================================', file=fp)
  print ('C   add up weighted layer inputs', file=fp)
  for i in range(n_out):
    out_name = make_tacs_name('{:s}_{:d}'.format(ltrE, i)) # this feeds the output layer biases
    i1 = 0
    ncards = 0  # for now, just assume there will be more than one
    last_out = ''
    plus = ''
    while i1 < n_hid:
      i2 = min(i1 + 9, n_hid)
      if i2 >= n_hid:
        this_out = out_name
      else:
        this_out = make_tacs_name('{:s}_{:d}{:s}'.format(ltrE, i, str(chr(ord('A')+ncards))))
      print ('{:s}{:s}  ={:s}{:s}{:s}'.format(grp, this_out, last_out, plus,
          '+'.join([make_tacs_name('{:s}{:d}_{:d}'.format(ltrD, i,j)) for j in range(i1,i2)])), file=fp)
      i1 += 9
      last_out = this_out
      plus = '+'
      ncards += 1
  print ('C ====================================================', file=fp)
  print ('C   output layer biases, net.2.bias', file=fp)
  vals = F['net.2.bias']
  for i in range(n_out):
    out_name = make_tacs_name('{:s}O{:d}'.format(tag, i))
    in_name = make_tacs_name('{:s}_{:d}'.format(ltrE, i))
    print ('{:s}{:s}  = {:12.9f}+{:s}'.format(grp, out_name, vals[i], in_name), file=fp)

if __name__ == '__main__':
  fname = 'hwpv3.atp'
  with open ('c:/src/pecblocks/examples/hwpv/tacs/tacs_fhf.json', 'r') as read_file:
    hwpv = json.load (read_file)
    print ('HWPV Model:', hwpv['name'], hwpv['type'], 'dt =', hwpv['t_step'])
    print ('  Inputs:', hwpv['COL_U'])
    print ('  Outputs:', hwpv['COL_Y'])
    F1 = hwpv['F1']
    F2 = hwpv['F2']
    H1s = hwpv['H1s']
    n_in1 = F1['n_in']
    n_hid1 = F1['n_hid']
    n_out1 = F1['n_out']
    n_a = H1s['na']
    n_b = H1s['nb']
    n_inh = H1s['n_in']
    n_outh = H1s['n_out']
    n_in2 = F2['n_in']
    n_hid2 = F2['n_hid']
    n_out2 = F2['n_out']
    print ('  F1: n_in={:d} n_hid={:d} n_out={:d} activation={:s}'.format (n_in1, n_hid1, n_out1, F1['activation']))
    print ('  H1s: n_in={:d} n_out={:d} n_a={:d} n_b={:d}'.format (n_inh, n_outh, n_a, n_b))
    print ('  F2: n_in={:d} n_hid={:d} n_out={:d} activation={:s}'.format (n_in2, n_hid2, n_out2, F2['activation']))

    fp = open (fname, 'w')
    print (TACS_template, file=fp)
    print ('C ================================================================', file=fp)
    print ('C calculate Vrms', file=fp)
    print ('90__PCCA', file=fp)
    print ('90__PCCB', file=fp)
    print ('90__PCCC', file=fp)
    print ('99____WC  =2*PI*____FC', file=fp)
    print ('99_THETA  =____WC*TIMEX', file=fp)
    print ('99_COSTH  =cos(_THETA)', file=fp)
    print ('99_COSTM  =cos(_THETA-2*PI/3)', file=fp)
    print ('99_COSTP  =cos(_THETA+2*PI/3)', file=fp)
    print ('99_SINTH  =sin(_THETA)', file=fp)
    print ('99_SINTM  =sin(_THETA-2*PI/3)', file=fp)
    print ('99_SINTP  =sin(_THETA+2*PI/3)', file=fp)
    print ('99____VD  =(__PCCA*_COSTH+__PCCB*_COSTM+__PCCC*_COSTP)/1.5', file=fp)
    print ('99____VQ  =-(__PCCA*_SINTH+__PCCB*_SINTM+__PCCC*_SINTP)/1.5', file=fp)
    print ('99____V0  =(__PCCA+__PCCB+__PCCC)/3.0', file=fp)
    print ('99__VRMS  =sqrt(1.5)*sqrt(____VD*____VD + ____VQ*____VQ)', file=fp)
    print ('99_GVRMS  =_____G*__VRMS/1000.0', file=fp)
    print ('C =============================================================================', file=fp)
    print ('C normalize the inputs', file=fp)
    idx = 0
    for key in hwpv['COL_U']:
      src_name = make_tacs_name(key.upper(), bTrack=False) # don't auto-make DUM variables, put some in DBM_start
      var_name = make_tacs_name('F1I{:d}'.format(idx))
      scale = float(hwpv['normfacs'][key]['scale'])
      offset = float(hwpv['normfacs'][key]['offset'])
      if offset < 0.0:
        sgn = '+'
      else:
        sgn = '-'
      print ('99{:s}  =({:s} {:s} {:.8f}) / {:.8f}'.format (var_name, src_name, sgn, abs(offset), scale), file=fp)
      idx += 1
    print ('C =============================================================================', file=fp)
    print ('C apply F1', file=fp)
    make_tacs_fblock (F=F1, tag='F1', ltr='A', grp='99', fp=fp)
    print ('C =============================================================================', file=fp)
    print ('C apply H1s', file=fp)
    for i in range(n_outh):
      for j in range(n_inh):
        den_key = 'a_{:d}_{:d}'.format(i,j)
        num_key = 'b_{:d}_{:d}'.format(i,j)
        H1s[num_key].reverse()
        H1s[den_key].reverse()
        out_name = make_tacs_name('H{:d}_{:d}'.format(i,j))
        in_name = make_tacs_name('F1O{:d}'.format(j))
        print ('{:2d}{:s}  +{:s}                                    1.0'.format(n_a-1, out_name, in_name), file=fp)
        print ('{:s}'.format(make_tacs_array(H1s[num_key])), file=fp)
        print ('{:s}'.format(make_tacs_array(H1s[den_key])), file=fp)
    for i in range(n_outh):
      out_name = make_tacs_name('F2I{:d}'.format(i))
      line = '98{:s}  ={:s}'.format(out_name,
            '+'.join([make_tacs_name('H{:d}_{:d}'.format(i,j)) for j in range(n_inh)]))
      print (line, file=fp)
    print ('C =============================================================================', file=fp)
    print ('C apply F2', file=fp)
    make_tacs_fblock (F=F2, tag='F2', ltr='J', grp='98', fp=fp)
    print ('C =============================================================================', file=fp)
    print ('C denormalize the outputs', file=fp)
    idx = 0
    for key in hwpv['COL_Y']:
      src_name = make_tacs_name(key.upper())
      var_name = make_tacs_name('F2O{:d}'.format(idx))
      scale = float(hwpv['normfacs'][key]['scale'])
      offset = float(hwpv['normfacs'][key]['offset'])
      if offset < 0.0:
        sgn = '-'
      else:
        sgn = '+'
      print ('98{:s}  ={:s} * {:.8f} {:} {:.8f}'.format (src_name, var_name, scale, sgn, abs(offset)), file=fp)
      idx += 1
    print ('C =============================================================================', file=fp)
    print ('C calculate current components from filter quantities', file=fp)
    print ('98_ARGDQ  =2.0*____WC*TIMEX', file=fp)
    print ('98__ARG0  =____WC*TIMEX', file=fp)
    print ('98____ID  =__IDLO + __IDHI*sin(_ARGDQ)', file=fp)
    print ('98____IQ  =__IQLO + __IQHI*sin(_ARGDQ)', file=fp)
    print ('98____I0  =__I0LO + __I0HI*sin(__ARG0)', file=fp)
    print ('C calculate current injections', file=fp)
    print ('98____IA  =____ID*_COSTH - ____IQ*_SINTH + ____I0', file=fp)
    print ('98____IB  =____ID*_COSTM - ____IQ*_SINTM + ____I0', file=fp)
    print ('98____IC  =____ID*_COSTP - ____IQ*_SINTP + ____I0', file=fp)
    print ('C request outputs', file=fp)
#   print ('98__OUT1  =___VDC', file=fp)
#   print ('98__OUT2  =___IDC', file=fp)
#   print ('98__OUT3  =____ID', file=fp)
#   print ('98__OUT4  =____IQ', file=fp)
#   print ('98__OUT5  =____I0', file=fp)
#   print ('98__OUT6  =____VD', file=fp)
#   print ('98__OUT7  =____VQ', file=fp)
#   print ('98__OUT8  =____V0', file=fp)
#   print ('98__OUT9  =__VRMS', file=fp)
#   print ('33__OUT1__OUT2__OUT3__OUT4__OUT5__OUT6__OUT7__OUT8__OUT9', file=fp)
    print ('33___VDC___IDC____ID__IDLO__IDHI____IQ__IQLO__IQHI__VRMS', file=fp)
    print ('33____VD____VQ____V0__I0LO__I0HI', file=fp)
    print ('C =============================================================================', file=fp)
    print (Network_template, file=fp)
    fp.close()

  quit() # DBM option doesn't work
  fname = 'hwpv3.dbm'
  fp = open (fname, 'w')
  print (DBM_start, file=fp)
  names = sorted(list(tacs_names))
  ntacs = len(names)
  i1 = 0
  ncards = 0
  nwritten = 0
  while i1 < ntacs: # and ncards < 210:
    i2 = min(i1 + 10, ntacs)
    print ('DUM {:s}'.format(','.join([elm for elm in names[i1:i2]])), file=fp)
    nwritten += (i2-i1)
    i1 += 10
    ncards += 1
  print ('$INCLUDE,HWPV3.ATP', file=fp)
  print (DBM_finish, file=fp)
  fp.close()
  print ('Needed {:d} dummy TACS variables, wrote {:d}'.format(ntacs, nwritten))

