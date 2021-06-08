import os
import sys
import torch
import pecblocks.util

root_path = './models/'

model_sets = [
   {'name':'test','blocks':'F_nl+G2'},
   {'name':'GtoVdc','blocks':'F1+G1+F2'},
   {'name':'GtoIdc','blocks':'F1+G1+F2'},
   {'name':'TtoVdc','blocks':'F1+G1+F2'},
   {'name':'TtoIdc','blocks':'F1+G1+F2'},
   {'name':'VdctoIdc','blocks':'F1+G1+F2'},
   {'name':'VdctoPac','blocks':'F1+G1+F2'},
   {'name':'VdctoQac','blocks':'F1+G1+F2'}
]

def format_tensor(t, dim):
    shape = list(t.shape)
    
    return str(t)
#    if t.shape[-1] > 1:
#        return ','.join('{:.5f}'.format(v) for v in t[0].tolist())
#    else:
#        return '{:.5f}'.format(t[0,0])

def slice_tensor(B, key, dim):
    t = B[key]
    shape = list(t.shape)
    print (key, shape, t)

def process_model_set (row):
    model_path = '{:s}{:s}/'.format(root_path, row['name'])
    model_type = row['blocks']
    blocks = model_type.split('+')
    print (model_path)
    for block in blocks:
        fname = '{:s}{:s}.pkl'.format (model_path, block)
        B = torch.load (fname)
        if 'G' in block:  # where is n_k?
            a = B['a_coeff']
            b = B['b_coeff']
            print ('  {:s} na={:d} nb={:d}'.format (block, a.shape[-1], b.shape[-1]))
            print ('       den=' + ','.join('{:.5f}'.format (v) for v in a[0,0].tolist()))
            print ('       num=' + ','.join('{:.5f}'.format (v) for v in b[0,0].tolist()))
        elif 'F' in block:  # assume tanh, how do we know nh?
            n0w = slice_tensor (B, 'net.0.weight', 0)
            quit()
            n0w = B['net.0.weight']
            n0b = B['net.0.bias']
            n2w = B['net.2.weight']
            n2b = B['net.2.bias']
            print ('  {:s} n0w{:s} n0b{:s} n2w{:s} n2b{:s}'.format (block, str(list(n0w.shape)), 
                                                                    str(list(n0b.shape)), 
                                                                    str(list(n2w.shape)), 
                                                                    str(list(n2b.shape))))
            print ('       n0w={:s}'.format (format_tensor(n0w,0)))
            print ('       n0b={:s}'.format (format_tensor(n0b,0)))
            print ('       n2w={:s}'.format (format_tensor(n2w,1)))
            print ('       n2b={:s}'.format (format_tensor(n2b,0)))
        else:
            print ('unrecognized block type {:s} for {:s}'.format (B, fname))

if __name__ == '__main__':
    for row in model_sets:
        process_model_set (row)

