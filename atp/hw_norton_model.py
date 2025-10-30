import os
import sys
import json
import numpy as np

model_sets = [
   {'name':'GtoIdc',   'blocks':'F1+G1+F2', 'atp_name':'GIDSRC'}
]

# these have to observe a FORTRAN 80-character line width
def zpolystr (key, np):
    cstr = '+\n      '.join('{:s}[{:d}]|z{:d}'.format (key,idx+1,-idx) for idx in range(np))
    return '({:s})'.format (cstr)

def write_constant_array (block, key, vec, fp):
    if len(vec) > 1:
        vstr = ',\n      '.join('{:.5f}'.format (v) for v in vec)
        print ('  {:s}_{:s}[1..{:d}] {{val:[{:s}]}}'.format (block, key, len(vec), vstr), file=fp)
    else:
        print ('  {:s}_{:s} {{val:{:.5f}}}'.format (block, key, vec[0]), file=fp)

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
    blk1 = 'F1'
    blk2 = 'G1'
    blk3 = 'F2'
    np1 = len(mdl[blk1]['n0b'])
    astr1 = '[1..{:d}]'.format(np1)
    nk2 = mdl[blk2]['n_k']
    na2 = len(mdl[blk2]['denominator']) + 1  # a0 always 1
    nb2 = len(mdl[blk2]['numerator']) + nk2  # first nk coefficients are zero
    astr2 = '[1..{:d}]'.format(na2)
    np3 = len(mdl[blk3]['n0b'])
    astr3 = '[1..{:d}]'.format(np3)

    fname = '{:s}.mod'.format(atp_name)
    fp = open (fname, 'w')

    print ('MODEL {:s} -- 6 character name limit'.format(atp_name), file=fp)
    write_norton_header (fp)
    print ('DATA  WC {dflt: 377.0}', file=fp)
    print ('VAR o1,o2,o3,i1', file=fp)
    print ('  {:s}_w1{:s} -- weighted inputs'.format(blk1, astr1), file=fp)
    print ('  {:s}_b1{:s} -- biased inputs'.format(blk1, astr1), file=fp)
    print ('  {:s}_th{:s} -- activation functions'.format(blk1, astr1), file=fp)
    print ('  {:s}_w2{:s} -- weighted outputs'.format(blk1, astr1), file=fp)
    print ('  {:s}_w1{:s} -- weighted inputs'.format(blk3, astr3), file=fp)
    print ('  {:s}_b1{:s} -- biased inputs'.format(blk3, astr3), file=fp)
    print ('  {:s}_th{:s} -- activation functions'.format(blk3, astr3), file=fp)
    print ('  {:s}_w2{:s} -- weighted outputs'.format(blk3, astr3), file=fp)

    print ('CONST', file=fp)
    write_constant_array (blk1, 'n0b', mdl[blk1]['n0b'], fp)
    write_constant_array (blk1, 'n0w', mdl[blk1]['n0w'], fp)
    write_constant_array (blk1, 'n2b', mdl[blk1]['n2b'], fp)
    write_constant_array (blk1, 'n2w', mdl[blk1]['n2w'], fp)

    a_coeffs = np.ones(na2)
    b_coeffs = np.zeros(nb2)
    a_coeffs[1:] = mdl[blk2]['denominator']
    b_coeffs[nk2:] = mdl[blk2]['numerator']
    write_constant_array (blk2, 'a', a_coeffs, fp)
    write_constant_array (blk2, 'b', b_coeffs, fp)

    write_constant_array (blk3, 'n0b', mdl[blk3]['n0b'], fp)
    write_constant_array (blk3, 'n0w', mdl[blk3]['n0w'], fp)
    write_constant_array (blk3, 'n2b', mdl[blk3]['n2b'], fp)
    write_constant_array (blk3, 'n2w', mdl[blk3]['n2w'], fp)

    print ('HISTORY o2 {dflt:0}', file=fp)
    print ('INIT', file=fp)
    print ('  {:s}_w1{:s} := 0'.format(blk1, astr1), file=fp)
    print ('  {:s}_b1{:s} := 0'.format(blk1, astr1), file=fp)
    print ('  {:s}_th{:s} := 0'.format(blk1, astr1), file=fp)
    print ('  {:s}_w2{:s} := 0'.format(blk1, astr1), file=fp)
    print ('  {:s}_w1{:s} := 0'.format(blk3, astr3), file=fp)
    print ('  {:s}_b1{:s} := 0'.format(blk3, astr3), file=fp)
    print ('  {:s}_th{:s} := 0'.format(blk3, astr3), file=fp)
    print ('  {:s}_w2{:s} := 0'.format(blk3, astr3), file=fp)
    print ('  o1 := 0', file=fp)
    print ('  o3 := 0', file=fp)
    print ('-- conductance', file=fp)
    print ('  g[1] := 0.000001  -- small shunt on the output terminal', file=fp)
    print ('  g[2] := 0.0       -- no coupling from output to G input', file=fp)
    print ('  g[3] := 0.0       -- no loading of the G input', file=fp)
    print ('ENDINIT', file=fp)
    print ('EXEC', file=fp)
    print ('-- initialize conductance', file=fp)
    print ('  if t=0 then', file=fp)
    print ('    flag:=1', file=fp)
    print ('  else', file=fp)
    print ('    flag:=0', file=fp)
    print ('  endif', file=fp)
    print ('-- F1 block', file=fp)
    print ('  i1 := v[2]', file=fp)
    print ('  {0:s}_w1{1:s} := i1 * {0:s}_n0w{1:s}'.format(blk1, astr1), file=fp)
    print ('  {0:s}_b1{1:s} := {0:s}_w1{1:s} + {0:s}_n0b{1:s}'.format(blk1, astr1), file=fp)
    print ('  {0:s}_th{1:s} := tanh({0:s}_b1{1:s})'.format(blk1, astr1), file=fp)
    print ('  FOR i:=1 to {0:d} DO {1:s}_w2[i] := {1:s}_th[i] * {1:s}_n2w[i] ENDFOR'.format(np1, blk1), file=fp)
    print ('  o1 := {0:s}_n2b'.format(blk1), file=fp)
    print ('  FOR i:=1 to {0:d} DO o1 := o1 + {1:s}_w2[i] ENDFOR'.format(np1, blk1), file=fp)

    print ('-- G1 block', file=fp)
    print ('  czfun(o2/o1) :=', file=fp)
    print (' {0:s}'.format (zpolystr ('{0:s}_b'.format(blk2), nb2)), file=fp)
    print ('  /', file=fp)
    print (' {0:s}'.format (zpolystr ('{0:s}_a'.format(blk2), na2)), file=fp)

    print ('-- F2 block', file=fp)
    print ('  {0:s}_w1{1:s} := o2 * {0:s}_n0w{1:s}'.format(blk3, astr3), file=fp)
    print ('  {0:s}_b1{1:s} := {0:s}_w1{1:s} + {0:s}_n0b{1:s}'.format(blk3, astr3), file=fp)
    print ('  {0:s}_th{1:s} := tanh({0:s}_b1{1:s})'.format(blk3, astr3), file=fp)
    print ('  FOR i:=1 to {0:d} DO {1:s}_w2[i] := {1:s}_th[i] * {1:s}_n2w[i] ENDFOR'.format(np3, blk3), file=fp)
    print ('  o3 := {0:s}_n2b'.format(blk3), file=fp)
    print ('  FOR i:=1 to {0:d} DO o3 := o3 + {1:s}_w2[i] ENDFOR'.format(np3, blk3), file=fp)

    print ('-- inject the Norton sources', file=fp)
    print ('  i[1] := o3', file=fp)
    print ('  is[1] := o3', file=fp)

    print ('ENDEXEC', file=fp)
    print ('ENDMODEL', file=fp)

    fp.close()

if __name__ == '__main__':
    with open ('models.json', 'r') as read_file:
        models = json.load (read_file)
    for row in model_sets:
        mdl = models[row['name']]
        atp_name = row['atp_name']
        write_one_atp_model (mdl, atp_name)

