import h5py
import sys, os
import numpy as np
from hdf_utils import hdf5_write
import pandas as pd


dest_h5_path = sys.argv[1]
src_h5_path = sys.argv[2]
fusion = int(sys.argv[3]) # if 1 then we concatenate features, if 0 then add to the dest_h5_path file
data_dir_tsv = sys.argv[4] # for the utt_id to look for in the case of fusion. Used only to get utt_list


if ',' in src_h5_path:
    src_h5_path = src_h5_path.split(',')
    src_h5_path = [i for i in src_h5_path if i != '']
else:
    src_h5_path = [src_h5_path]
 
if not fusion:
    # same list of utt_id need not be there in every h5 considered   
    dest_h5 = h5py.File(dest_h5_path, 'w')
    for src_h5_temp_path in src_h5_path:
        src_h5_temp_path = os.path.abspath(src_h5_temp_path)
        print(src_h5_temp_path)
        src_h5_temp = h5py.File(src_h5_temp_path, 'r')
        for utt_name in src_h5_temp:
            print(utt_name)
            dest_h5[utt_name] = h5py.ExternalLink(src_h5_temp_path, utt_name)
        src_h5_temp.close()
else:
    # same list of utt_id need to there in every h5 considered
    # dest_h5_path is NOT considered here
    # data_dir_tsv must exist and used for utt_id list
    utt_list = pd.read_csv(data_dir_tsv, sep=',', header=None)
    utt_list = list(utt_list[0])  

    src_h5_feats = {}
    for ind,src_h5_temp in enumerate(src_h5_path):
        if src_h5_temp.endswith('.scp_h5'):
            temp = pd.read_csv(src_h5_temp, sep=',', header=None)
            temp = list(set(temp[1]))
            if len(temp) > 1:
                raise ValueError(f'currently having multiple h5 files is not supported in a scp_h5 file')
            else:
                src_h5_temp = temp[0]
        print(src_h5_temp)
        src_h5_feats[ind] = h5py.File(src_h5_temp, 'r')


    dest_h5_path = os.path.abspath(dest_h5_path)
    out_scp_h5_path = dest_h5_path.split('.h5')[0] + '.scp_h5'
    with open(out_scp_h5_path, 'a') as f:
        for utt_id in utt_list:
            utt_feats = []
            for ind in range(len(src_h5_path)):
                feats = src_h5_feats[ind][utt_id][()]
                if len(feats.squeeze().shape) == 1:
                    utt_feats.append(feats)
                else:
                    utt_feats.append(np.mean(feats, axis=0).squeeze())

            #import pdb; pdb.set_trace()                    
            try:
                concat_feats = np.concatenate(utt_feats, axis=-1)
            except:
                concat_feats = np.vstack(utt_feats).squeeze()
            print(f'concatenated features shape is {concat_feats.shape}')
            hdf5_write(concat_feats, dest_h5_path, utt_id)
            f.write(utt_id + ',' + dest_h5_path + '\n')    





