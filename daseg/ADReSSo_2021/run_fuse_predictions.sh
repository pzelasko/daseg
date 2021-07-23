

#############################3

## this script is used to fuse predictions from various models
## first it writes all folds into one file 
## then use that for fusion of several models


##################################


dir_suffix=$1 #x_vectors_linear_models_diagnosis
linear_model=$2 #lr
fused_features_path=$3 # fused_features/fusion_all_preds.h5 fused_features/fusion_all_preds_MMSE.h5
stage=$4

### combine all CV predictions and write to a file for each expt. Then fuse the features into one vector from all experiments

if [[ $stage == 1 ]]
then
    #for features_type in concat_ResNet_fs100ms_SpeechBrain_spkr_verif_feats concat_ResNet_fs10ms_SpeechBrain_spkr_verif_feats ResNet_clean_xvector_feats ResNet_clean_xvector_feats_fs100ms concat_ResNet_clean_fs100ms_SpeechBrain_spkr_verif_feats vggish_features speechbrain_EncDec_features_ASR prosody_features_20msec prosody_features_30msec ResNet_aug_xvector_feats_fs100ms ResNet_aug_xvector_feats speechbrain_spkr_verif_feats fusion_all_features
    #do

    model_seed=42 # 43 44
    for transcript_version in v2 v3 v4 v5 v6 v7
    do

        expt_dir=/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_${cv}_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_${transcript_version}/bert-base-uncased_ADReSSo_${model_seed}/
        src_h5_path=
        for cv in $(seq 1 10)
        do
            src_h5_path=${src_h5_path},/export/fs03/a06/rpapagari/ADReSSo_IS2021_expts/ADReSSo_CV_${cv}_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_${transcript_version}/bert-base-uncased_ADReSSo_${model_seed}/resultschallenge_test.h5
        done
        dest_h5_path=${expt_dir}/preds_test_all_cv.h5
        
        echo $src_h5_path
        
        python scripts/concat_h5_v2.py $dest_h5_path $src_h5_path 0 temp
    done
fi

###########################################


if [[ $stage == 2 ]]
then
    data_dir_tsv=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest/all.tsv
    src_h5_path=
    for features_type in ResNet_clean_xvector_feats_fs100ms speechbrain_EncDec_features_ASR ResNet_clean_xvector_feats ResNet_aug_xvector_feats_fs100ms
#concat_ResNet_fs100ms_SpeechBrain_spkr_verif_feats concat_ResNet_fs10ms_SpeechBrain_spkr_verif_feats ResNet_clean_xvector_feats ResNet_clean_xvector_feats_fs100ms concat_ResNet_clean_fs100ms_SpeechBrain_spkr_verif_feats vggish_features speechbrain_EncDec_features_ASR prosody_features_20msec prosody_features_30msec ResNet_aug_xvector_feats_fs100ms ResNet_aug_xvector_feats speechbrain_spkr_verif_feats fusion_all_features
    do
        expt_dir=${dir_suffix}/${features_type}/${linear_model}
        src_h5_path=${src_h5_path},${expt_dir}/preds_test_all_cv.h5
    done
    
    dest_h5_path=$fused_features_path
    echo $src_h5_path
    python scripts/concat_h5_v2.py $dest_h5_path $src_h5_path 1 $data_dir_tsv
fi


