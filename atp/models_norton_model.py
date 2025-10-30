import os
import sys
import json
import numpy as np

# these have to observe a FORTRAN 80-character line width
def zpolystr (key, np):
  cstr = '+\n    '.join('{:s}[{:d}]|z{:d}'.format (key,idx+1,-idx) for idx in range(np))
  return '({:s})'.format (cstr)

def write_constant_array (key, vec, fp):
  if type(vec[0]) == list:
    flat = [element for sublist in vec for element in sublist]
  else:
    flat = vec
  vstr = ',\n   '.join('{:12.8f}'.format (v) for v in flat)
  print ('  {:s}[1..{:d}] {{val:[{:s}]}}'.format(key, len(flat), vstr), file=fp)

def write_norton_header (fp):
  print ("""-- Start header. Do not modify the type-94 header. 
comment---------------------------------------------------------------
 | First, declarations required for any type 94 Norton non-TR model    |
 | - these data and input values are provided to the model by ATP      |
 | - these output values are used by ATP                               |
 | - these names can be changed, except 'n', but not their order       |
 -------------------------------------------------------------endcomment

DATA  n                      -- number of phases
      ng {dflt: n*(n+1)/2}   -- number of conductances on each side

INPUT v[1..n]   -- voltage(t) at each left node
      v0[1..n]  -- voltage(t=0) at each left node
      i0[1..n]  -- current(t=0) into each left node

VAR   i[1..n]   -- current(t) into each left node (for plotting)
      is[1..n]  -- Norton source(t+timestep) at each left node
      g[1..ng]  -- conductance(t+timestep) at each left node
                -- sequence is 1-gr, 1-2, 1-3..1-n,2-gr,2-3..2-n,...n-gr
      flag      -- set to 1 whenever conductance value is modified

OUTPUT i[1..n],is[1..n],g[1..ng],flag

 comment---------------------------------------------------------------
 | Next, declarations of user-defined data for this particular model   |
 | - their value is defined at the time of using the type-94 component |
 -------------------------------------------------------------endcomment
-- End header.  """, file=fp)

def write_one_atp_model (mdl, atp_name):
  F1 = mdl['F1']
  H1 = mdl['H1']
  F2 = mdl['F2']
  nF1i = F1['n_in']
  nF1h = F1['n_hid']
  nF1o = F1['n_out']
  nF2i = F2['n_in']
  nF2h = F2['n_hid']
  nF2o = F2['n_out']
  na = H1['n_a']
  nb = H1['n_b']
  nk = H1['n_k']
  nH1i = H1['n_in']
  nH1o = H1['n_out']

  fname = '{:s}.mod'.format(atp_name)
  fp = open (fname, 'w')

  print ('MODEL {:s} -- 6 character name limit'.format(atp_name), file=fp)
  write_norton_header (fp)
  print ('DATA  WC {dflt: 377.0}', file=fp)
  print ('VAR', file=fp)
  print ('  F1_0w[1..{:d}] -- weighted inputs'.format(nF1h*nF1i), file=fp)
  print ('  F1_0b[1..{:d}] -- biased inputs'.format(nF1h), file=fp)
  print ('  F1_th [1..{:d}] -- activation functions'.format(nF1h), file=fp)
  print ('  F1_2w[1..{:d}] -- weighted outputs'.format(nF1o*nF1h), file=fp)
  print ('  F1_2b[1..{:d}] -- biased outputs'.format(nF1o), file=fp)
  print ('  F2_0w[1..{:d}] -- weighted inputs'.format(nF2h*nF2i), file=fp)
  print ('  F2_0b[1..{:d}] -- biased inputs'.format(nF2h), file=fp)
  print ('  F2_th [1..{:d}] -- activation functions'.format(nF2h), file=fp)
  print ('  F2_2w[1..{:d}] -- weighted outputs'.format(nF2o*nF2h), file=fp)
  print ('  F2_2b[1..{:d}] -- biased outputs'.format(nF2o), file=fp)

  print ('CONST', file=fp)
  write_constant_array ('F1_n0w', F1['net.0.weight'], fp)
  write_constant_array ('F1_n0b', F1['net.0.bias'], fp)
  write_constant_array ('F1_n2w', F1['net.2.weight'], fp)
  write_constant_array ('F1_n2b', F1['net.2.bias'], fp)
  write_constant_array ('F2_n0w', F2['net.0.weight'], fp)
  write_constant_array ('F2_n0b', F2['net.0.bias'], fp)
  write_constant_array ('F2_n2w', F2['net.2.weight'], fp)
  write_constant_array ('F2_n2b', F2['net.2.bias'], fp)
  for i in range(nH1o):
    for j in range(nH1i):
      key = '{:d}_{:d}'.format(i, j)
      vec = H1['a_'+key]
      vec.insert(0,1.0)
      write_constant_array ('Ha'+key, vec, fp)
      vec = H1['b_'+key]
      vec.insert(0,0.0)
      write_constant_array ('Hb'+key, vec, fp)

  print ('HISTORY o2 {dflt:0}', file=fp)
  print ('INIT', file=fp)
  print ('  F1_0w[1..{:d}] := 0'.format(nF1h*nF1i), file=fp)
  print ('  F1_0b[1..{:d}] := 0'.format(nF1h), file=fp)
  print ('  F1_th [1..{:d}] := 0'.format(nF1h), file=fp)
  print ('  F1_2w[1..{:d}] := 0'.format(nF1o*nF1h), file=fp)
  print ('  F1_2b[1..{:d}] := 0'.format(nF1o), file=fp)
  print ('  F2_0w[1..{:d}] := 0'.format(nF2h*nF2i), file=fp)
  print ('  F2_0b[1..{:d}] := 0'.format(nF2h), file=fp)
  print ('  F2_th [1..{:d}] := 0'.format(nF2h), file=fp)
  print ('  F2_2w[1..{:d}] := 0'.format(nF2o*nF2h), file=fp)
  print ('  F2_2b[1..{:d}] := 0'.format(nF2o), file=fp)
  print ('  o1 := 0', file=fp)
  print ('  o3 := 0', file=fp)
  print ('-- conductance', file=fp)
  print ('  g[1] := 0.000001  -- small shunt on the output terminal', file=fp)
  print ('  g[2] := 0.0     -- no coupling from output to G input', file=fp)
  print ('  g[3] := 0.0     -- no loading of the G input', file=fp)
  print ('ENDINIT', file=fp)
  print ('EXEC', file=fp)
  print ('-- initialize conductance', file=fp)
  print ('  if t=0 then', file=fp)
  print ('    flag:=1', file=fp)
  print ('  else', file=fp)
  print ('    flag:=0', file=fp)
  print ('  endif', file=fp)
#   print ('-- F1 block', file=fp)
#   print ('  i1 := v[2]', file=fp)
#   print ('  {0:s}_w1{1:s} := i1 * {0:s}_n0w{1:s}'.format(blk1, astr1), file=fp)
#   print ('  {0:s}_b1{1:s} := {0:s}_w1{1:s} + {0:s}_n0b{1:s}'.format(blk1, astr1), file=fp)
#   print ('  {0:s}_th{1:s} := tanh({0:s}_b1{1:s})'.format(blk1, astr1), file=fp)
#   print ('  FOR i:=1 to {0:d} DO {1:s}_w2[i] := {1:s}_th[i] * {1:s}_n2w[i] ENDFOR'.format(np1, blk1), file=fp)
#   print ('  o1 := {0:s}_n2b'.format(blk1), file=fp)
#   print ('  FOR i:=1 to {0:d} DO o1 := o1 + {1:s}_w2[i] ENDFOR'.format(np1, blk1), file=fp)
#
#   print ('-- G1 block', file=fp)
#   print ('  czfun(o2/o1) :=', file=fp)
#   print (' {0:s}'.format (zpolystr ('{0:s}_b'.format(blk2), nb2)), file=fp)
#   print ('  /', file=fp)
#   print (' {0:s}'.format (zpolystr ('{0:s}_a'.format(blk2), na2)), file=fp)
#
#   print ('-- F2 block', file=fp)
#   print ('  {0:s}_w1{1:s} := o2 * {0:s}_n0w{1:s}'.format(blk3, astr3), file=fp)
#   print ('  {0:s}_b1{1:s} := {0:s}_w1{1:s} + {0:s}_n0b{1:s}'.format(blk3, astr3), file=fp)
#   print ('  {0:s}_th{1:s} := tanh({0:s}_b1{1:s})'.format(blk3, astr3), file=fp)
#   print ('  FOR i:=1 to {0:d} DO {1:s}_w2[i] := {1:s}_th[i] * {1:s}_n2w[i] ENDFOR'.format(np3, blk3), file=fp)
#   print ('  o3 := {0:s}_n2b'.format(blk3), file=fp)
#   print ('  FOR i:=1 to {0:d} DO o3 := o3 + {1:s}_w2[i] ENDFOR'.format(np3, blk3), file=fp)
#
  print ('-- inject the Norton sources', file=fp)
  print ('  i[1] := o3', file=fp)
  print ('  is[1] := o3', file=fp)

  print ('ENDEXEC', file=fp)
  print ('ENDMODEL', file=fp)

  fp.close()

if __name__ == '__main__':
  with open ('c:/src/pecblocks/examples/hwpv/models/unbalanced_fhf.json', 'r') as read_file:
    mdl = json.load (read_file)
    write_one_atp_model (mdl, 'HWPV3')

