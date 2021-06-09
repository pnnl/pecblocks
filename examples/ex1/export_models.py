import os
import sys
import torch
#import pecblocks.util
import json

root_path = './models/'

model_sets = [
   {'name':'test','blocks':'F_nl+G2','n_k':1},
   {'name':'GtoVdc','blocks':'F1+G1+F2','n_k':1},
   {'name':'GtoIdc','blocks':'F1+G1+F2','n_k':1},
   {'name':'TtoVdc','blocks':'F1+G1+F2','n_k':1},
   {'name':'TtoIdc','blocks':'F1+G1+F2','n_k':1},
   {'name':'VdctoIdc','blocks':'F1+G1+F2','n_k':0},
   {'name':'VdctoPac','blocks':'F1+G1+F2','n_k':0},
   {'name':'VdctoQac','blocks':'F1+G1+F2','n_k':0}
]

def slice_tensor(B, key, idx=0):
    t = B[key]
    shape = list(t.shape)
    ndim = len(shape)
    if ndim == 1:
        return t[:].numpy()
    elif ndim == 2:
        if idx == 0:
            return t[:,0].numpy()
        elif idx == 1:
            return t[0,:].numpy()
        else:
            print (key, shape, 'unsupported index', idx)
    elif ndim == 3:
        if idx == 0:
            return t[:,0,0].numpy()
        elif idx == 1:
            return t[0,:,0].numpy()
        elif idx == 2:
            return t[0,0,:].numpy()
        else:
            print (key, shape, 'unsupported index', idx)
    else:
        print (key, shape, 'too many dimensions')
    return None

def process_model_set (row):
    model_path = '{:s}{:s}/'.format(root_path, row['name'])
    model_type = row['blocks']
    blocks = model_type.split('+')
    print (model_path)
    model = {'name':row['name'],'type':row['blocks']}
    for block in blocks:
        fname = '{:s}{:s}.pkl'.format (model_path, block)
        B = torch.load (fname)
        if 'G' in block:  # where is n_k?
            a = slice_tensor (B, 'a_coeff', 2)
            b = slice_tensor (B, 'b_coeff', 2)
            model[block] = {'numerator':b.tolist(), 'denominator':a.tolist(), 'n_k':row['n_k']}
#           print (B.keys())
#           print ('  {:s} na={:d} nb={:d}'.format (block, a.shape[-1], b.shape[-1]))
#           print ('       den=' + ','.join('{:.4f}'.format (v) for v in a))
#           print ('       num=' + ','.join('{:.4f}'.format (v) for v in b))
        elif 'F' in block:  # assume tanh, how do we know nh?
            n0w = slice_tensor (B, 'net.0.weight')
            n0b = slice_tensor (B, 'net.0.bias')
            n2w = slice_tensor (B, 'net.2.weight', 1)
            n2b = slice_tensor (B, 'net.2.bias')
            model[block] = {'n0w':n0w.tolist(), 'n0b':n0b.tolist(), 'n2w':n2w.tolist(), 'n2b':n2b.tolist()}
#           print ('  {:s} {:d} {:d}'.format (block, len(n0w), len(n2b)))
#           print ('       n0w=' + ','.join('{:.4f}'.format (v) for v in n0w))
#           print ('       n0b=' + ','.join('{:.4f}'.format (v) for v in n0b))
#           print ('       n2w=' + ','.join('{:.4f}'.format (v) for v in n0w))
#           print ('       n2b=' + ','.join('{:.4f}'.format (v) for v in n2b))
        else:
            print ('unrecognized block type {:s} for {:s}'.format (B, fname))
    return model

if __name__ == '__main__':
    models = {}
    for row in model_sets:
        models[row['name']] = process_model_set (row)
    with open ('models.json', 'w') as write_file:
        json.dump (models, write_file, indent=4, sort_keys=True)


