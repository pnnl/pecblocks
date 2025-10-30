import os
import sys
import json
import numpy as np

model_sets = [
   {'name':'GtoVdc',   'blocks':'F1+G1+F2', 'atp_name':'GTOVDC'},
   {'name':'GtoIdc',   'blocks':'F1+G1+F2', 'atp_name':'GTOIDC'},
   {'name':'TtoVdc',   'blocks':'F1+G1+F2', 'atp_name':'TTOVDC'},
   {'name':'TtoIdc',   'blocks':'F1+G1+F2', 'atp_name':'TTOIDC'},
   {'name':'VdctoIdc', 'blocks':'F1+G1+F2', 'atp_name':'VTOIDC'},
   {'name':'VdctoPac', 'blocks':'F1+G1+F2', 'atp_name':'VTOPAC'},
   {'name':'VdctoQac', 'blocks':'F1+G1+F2', 'atp_name':'VTOQAC'}
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
#    print ('DATA', file=fp)
    print ('INPUT i1', file=fp)
    print ('OUTPUT o1,o2,o3', file=fp)
    print ('VAR o1,o2,o3', file=fp)
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
    print ('ENDINIT', file=fp)
    print ('EXEC', file=fp)
    print ('-- F1 block', file=fp)
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

    print ('ENDEXEC', file=fp)
    print ('ENDMODEL', file=fp)

    fp.close()

if __name__ == '__main__':
#   if len(sys.argv) > 3:
#       json_file = sys.argv[1]
#       model_name = sys.argv[2]
#       atp_name = sys.argv[3]
#   else:
#       json_file = 'models.json'
#       model_name = 'GtoVdc'
#       atp_name = 'GTOVDC'  # should be <= 6 upper-case characters
    with open ('models.json', 'r') as read_file:
        models = json.load (read_file)
#   if model_name not in models:
#       print ('Model {:s} was not found in {:s}'.format (model_name, json_file))
#       quit()
    for row in model_sets:
        mdl = models[row['name']]
        atp_name = row['atp_name']
        write_one_atp_model (mdl, atp_name)

