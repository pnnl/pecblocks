import h5py

PREFIX = 'scope'
input_path = 'd:/data/'
output_path = 'c:/data/sdi3rc.hdf5'
output_path = 'c:/data/sdi3merged.hdf5'
output_path = 'd:/data/sdi5.hdf5'

if __name__ == '__main__':
  print ('Merging HWPV training records {:s}'.format (output_path))
  f_out = h5py.File (output_path, 'w')
  idx = 0

  for root in ['sdi5a', 'sdi5b']: # ['sdi3', 'sdi3rc', 'sdi3uq']:
    filename = '{:s}{:s}.hdf5'.format (input_path, root)
    print ('Reading', filename)
    with h5py.File(filename, 'r') as f_in:
      for grp_name, grp_in in f_in.items():
        new_name = '{:s}{:d}'.format (PREFIX, idx)
        idx += 1
        print ('  copying {:s} to {:s}'.format(grp_name, new_name))
        grp_out = f_out.create_group (new_name)
        for tag in ['t', 'Fc', 'Ud', 'Uq', 'Rc', 'Vdc', 'Idc', 'Vd', 'Vq', 'Vrms', 'Id', 'Iq', 'Irms']:
          grp_out.create_dataset (tag, data=grp_in[tag], compression='gzip')

  f_out.close()

