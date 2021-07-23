import h5py
import sys, os
import numpy as np
from hdf_utils import hdf5_write
import pandas as pd
import glob


def concat_h5(dest_h5_path, src_h5_path, fusion, data_dir_tsv):
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
            src_h5_temp = h5py.File(src_h5_temp_path, 'r')
            for utt_name in src_h5_temp:
                #dest_h5[utt_name] = h5py.ExternalLink(src_h5_temp_path, utt_name)
                dest_h5[utt_name] = src_h5_temp[utt_name][()]
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
                    if np.isscalar(feats):
                        feats = np.array([feats])
                        utt_feats.append(feats)
                    else:
                        if len(feats.squeeze().shape) == 1:
                            utt_feats.append(feats)
                        else:
                            utt_feats.append(np.mean(feats, axis=0).squeeze())
    
                #import pdb; pdb.set_trace()                    
                try:
                    concat_feats = np.concatenate(utt_feats, axis=-1)
                except:
                    concat_feats = np.vstack(utt_feats).squeeze()
                import pdb; pdb.set_trace()
                print(f'concatenated features shape is {concat_feats.shape}')
                hdf5_write(concat_feats, dest_h5_path, utt_id)
                f.write(utt_id + ',' + dest_h5_path + '\n')    


task_type = '_MMSE' #''
task_type = ''


########## for text models
for model_seed in [42, 43, 44]:
    for transcript_version in ['v2', 'v3', 'v4', 'v5', 'v6', 'v7']:
        expt_dir_part1 = '/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_'
        expt_dir_part2 = '_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_'
        expt_dir_part3 =  task_type + '/bert-base-uncased_ADReSSo_'
        expt_dir_part4 = '/results.h5'
    
        src_h5_path=[]
        for cv in range(1, 11):
            temp = expt_dir_part1 + str(cv) + expt_dir_part2 + transcript_version + expt_dir_part3 + str(model_seed) + expt_dir_part4
            src_h5_path.append(temp)
        src_h5_path = ','.join(src_h5_path)
        print(src_h5_path)
        dest_h5_path = expt_dir_part1 + str(1) + expt_dir_part2 + transcript_version + expt_dir_part3 + str(model_seed) + '/preds_test_all_cv.h5'
        fusion = False
        data_dir_tsv = 'temp'
    
        concat_h5(dest_h5_path, src_h5_path, fusion, data_dir_tsv)

############# for speech models 
for model_seed in [42, 43, 44]:
    for concat_aug in [-1]:
        expt_dir_part1 = '/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_'
        expt_dir_part2 = '_ExactLabelScheme_smoothSegmentation'
        expt_dir_part4 = '/results.h5'
    
        frame_len = 0.1
        xvector_model_id = '_SoTAEnglishXVector'
        xvector_model_id = '_SoTAEnglishXVectorClean'
        pre_trained_model = 'True'
        pre_train_suffix = xvector_model_id + pre_trained_model
        suffix = pre_train_suffix + '_frame_len' + str(frame_len) + '_UttClassif_bs6_gacc5_concat_aug_' + str(concat_aug) + '_model_TrainDevTest_maxlen_8000_Epochs_50' + task_type + '/resnet_SeqClassification_ADReSSo_' + str(model_seed)

        src_h5_path = expt_dir_part1 + '*' + expt_dir_part2 + suffix + expt_dir_part4 
        src_h5_path = glob.glob(src_h5_path)
        src_h5_path = ','.join(src_h5_path) 
        dest_h5_path = expt_dir_part1 + '1' + expt_dir_part2 + suffix + '/preds_test_all_cv.h5' 

        #print(src_h5_path)
        print(dest_h5_path)
        fusion = False
        data_dir_tsv = 'temp'
    
        concat_h5(dest_h5_path, src_h5_path, fusion, data_dir_tsv)



for model_seed in [42, 43, 44]:
    best_models_speech = '/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest_maxlen_8000_Epochs_50' + task_type + '/resnet_SeqClassification_ADReSSo_' + str(model_seed) + '/preds_test_all_cv.h5'

    best_models_transcript = '/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7' + task_type + '/bert-base-uncased_ADReSSo_' + str(model_seed) + '/preds_test_all_cv.h5'

    src_h5_path = best_models_speech + ',' + best_models_transcript
    dest_h5_path = 'fused_features/' + 'SoTAEnglishXVectorClean_frame_len0.1_concat_aug_-1_seed_' + str(model_seed) + '_' + '_BERT_transcripts_v7' + task_type + '.h5'
    fusion = True
    data_dir_tsv = '/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest/all.tsv'

    concat_h5(dest_h5_path, src_h5_path, fusion, data_dir_tsv)




