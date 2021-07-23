
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
use_grid=${8-True}
task_name=${9-ConvClassif_Longformer}

######### paramters previously set as default.Now, you would like to write them per task to avoid danger of bugs. Keeping it here as just a reference 
#corpus=IEMOCAP_v2 #_CV_${cv}
#no_epochs=50
#max_sequence_length=512
#frame_len=0.1
#results_suffix=.pkl
#model_name=longformer
#pre_trained_model=False
#test_file=test.tsv
#############################

emospotloss_wt=-100
emospot_concat=False
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/journal_v2
label_smoothing_alpha=0
full_speech=False
classwts=1

##########
# you should have tasks for longformer, bilstm, ResNet, xformer, 2-stage training of ResNet and longformer
# uttclassif, conversations
# IEMOCAP, SWBD
# 

#corpus, max_sequence_length, model_name, pre_trained_model, frame_len, data_dir, no_epochs, suffix, test_file, results_suffix

######################## convclassif    #############################

if [[ $task_name == IEMOCAP_UttClassif_ResNet_longformer_2stages ]]; then
## convclassif with x-vector features   #####
    corpus=IEMOCAP_v2
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    xvector_model_id=SoTAEnglishXVector_IsolatedUttTraining_run1
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/xvectorfeat_data_dirs/${xvector_model_id}/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    no_epochs=100
    frame_len=0.08
    suffix=${xvector_model_id}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == IEMOCAP_ConvClassif_Longformer ]]; then
    corpus=IEMOCAP_v2
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
    no_epochs=100
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == IEMOCAP_ConvClassif_Longformer_spkrind ]]; then
    corpus=IEMOCAP_v2
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
    no_epochs=100
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == IEMOCAP_ConvClassif_bilstm ]]; then
    corpus=IEMOCAP_v2
    max_sequence_length=512
    model_name=bilstm
    pre_trained_model=False
    frame_len=0.1
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
    no_epochs=100
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_debug
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == IEMOCAP_ConvClassif_ResNet ]]; then
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    no_epochs=100
    frame_len=0.01
    model_name=ResNet
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}
    
    
    ###################### convclassif with smoothed_overlap_silence_OOS    #############################
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOS_clean_noise123_music123/cv_${cv}
    #no_epochs=100
    ##suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_smoothed_overlap_silence_OOS
    ##results_suffix=.pkl
    #
    #label_smoothing_alpha=0.05 # 0.1
    #suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_smoothed_overlap_silence_OOS_labelsmoothing${label_smoothing_alpha}
    #
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms_clean_noise123_music123/cv_${cv}
    #results_suffix=_ASHNF_UttEval_fs100ms.pkl
fi



if [[ $task_name == IEMOCAP_ConvClassif_xformer_NoInfreqEmo ]]; then
####################  convclassif with xformer and fs=10ms  ######################
    corpus=IEMOCAP_v2
    no_epochs=100
    frame_len=0.01
    model_name=xformer
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    test_file=test.tsv
    results_suffix=_with_uttids.pkl

    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}

    if [[ $eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
        results_suffix=_ASHNF_ExciteMapHap_fs10ms_UttEval.pkl
    fi
fi

 if [[ $task_name == IEMOCAP_ConvClassif_xformer_smoothed_overlap_silenceNone_OOSNone_spkrind ]]; then
####################  convclassif with xformer and fs=10ms  ######################
    corpus=IEMOCAP_v2
    no_epochs=100
    frame_len=0.01
    model_name=xformer
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    test_file=test.tsv
    results_suffix=_with_uttids.pkl

    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_spkrind
    #full_speech=True
    results_suffix=_with_uttids.pkl


    ################# only for evaluation
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrM_fs10ms_clean_noise123_music123/cv_${cv}
    results_suffix=_with_uttids_SpkrM.pkl
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrF_fs10ms_clean_noise123_music123/cv_${cv}
    results_suffix=_with_uttids_SpkrF.pkl
    #############
 
fi

if [[ $task_name == IEMOCAP_ConvClassif_xformer_smoothed_overlap_silenceNone_OOSNone ]]; then
####################  convclassif with xformer and fs=10ms  ######################
    corpus=IEMOCAP_v2
    no_epochs=100
    frame_len=0.01
    model_name=xformer
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    test_file=test.tsv
    results_suffix=_with_uttids.pkl

    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_fs10ms_clean_noise123_music123/cv_${cv}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone
    ##full_speech=True
    results_suffix=_with_uttids.pkl
    
    ################# only for evaluation
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrM_fs10ms_clean_noise123_music123/cv_${cv}
    results_suffix=_with_uttids_SpkrM.pkl
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrF_fs10ms_clean_noise123_music123/cv_${cv}
    results_suffix=_with_uttids_SpkrF.pkl
    ###############
fi

if [[ $task_name == IEMOCAP_ConvClassif_xformer_cnnop_segpool_smoothed_overlap_silence_OOSNone_spkrind ]]; then
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    corpus=IEMOCAP_v2
    no_epochs=100
    frame_len=0.01
    model_name=xformer_cnnop_segpool #xformersegpool
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    results_suffix=_with_uttids.pkl

    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments_debug
    test_file=test.tsv
    results_suffix=_with_uttids.pkl

    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
    #suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_spkrind_PoolSegments
    #results_suffix=_with_uttids.pkl
fi 


######################   uttclassif   #############################
if [[ $task_name == IEMOCAP_UttClassif_longformer ]]; then
    corpus=IEMOCAP_v2
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    no_epochs=100

    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms_clean_noise123_music123/cv_${cv}
    
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps
    results_suffix=.pkl

    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
    #results_suffix=_conversations.pkl
fi

if [[ $task_name == IEMOCAP_UttClassif_bilstm ]]; then
    ##data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms/cv_${cv}
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms_clean_noise123_music123/cv_${cv}
    corpus=IEMOCAP_v2
    no_epochs=100
    max_sequence_length=512
    frame_len=0.1
    pre_trained_model=False
    model_name=bilstm    
    
    suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}
    results_suffix=.pkl
    
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
    results_suffix=_conversations.pkl
fi

if [[ $task_name == IEMOCAP_UttClassif_ResNet ]]; then
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    corpus=IEMOCAP_v2
    no_epochs=100
    frame_len=0.01
    model_name=ResNet
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}
    
    data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    results_suffix=_conversations.pkl
fi

################# 
if [[ $task_name == MELD_Emotion_UttClassif_bilstm_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_Emotion_8k_fs10ms_allutts/
    corpus=MELD_Emotion
    no_epochs=50
    frame_len=0.01
    max_sequence_length=300 # 4000  #2048
    test_file=test.tsv
    model_name=bilstm_SeqClassification
    pre_trained_model=False
    
    suffix=_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_seqclassif 
    results_suffix=_with_uttids.pkl
fi

if [[ $task_name == MELD_Emotion_UttClassif_longformer_seqclassif ]]
then
    corpus=MELD_Emotion
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_Emotion_8k_fs10ms_allutts/
    no_epochs=50
    frame_len=0.01
    max_sequence_length=512 # 4000  #2048
    test_file=test.tsv
    model_name=longformer_speech_SeqClassification
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_seqclassif 
    results_suffix=_with_uttids.pkl
fi

if [[ $task_name == MELD_Sentiment_UttClassif_bilstm_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_Sentiment_8k_fs10ms_allutts/
    corpus=MELD_Sentiment
    no_epochs=50
    frame_len=0.01
    max_sequence_length=300 # 4000  #2048
    test_file=test.tsv
    model_name=bilstm_SeqClassification
    pre_trained_model=False
    
    suffix=_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_seqclassif_MELDSentiment
    results_suffix=_with_uttids.pkl
fi

if [[ $task_name == MELD_Sentiment_UttClassif_longformer_seqclassif ]]
then
    corpus=MELD_Sentiment
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_Sentiment_8k_fs10ms_allutts/
    no_epochs=50
    frame_len=0.01
    max_sequence_length=512 # 4000  #2048
    test_file=test.tsv
    model_name=longformer_speech_SeqClassification
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_seqclassif_MELDSentiment
    results_suffix=_with_uttids.pkl
fi


######################## convclassif    #############################
if [[ $task_name == MELD_Emotion_ConvClassif_Longformer_fs10ms_spkrind ]]; then
    corpus=MELD_Emotion
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.01
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs10ms_clean_noise123_music123
    no_epochs=100
    suffix=_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_spkrind_MELD_Emotion
    test_file=test.tsv
    results_suffix=.pkl
fi


if [[ $task_name == MELD_Emotion_ConvClassif_bilstm_fs10ms_spkrind ]]; then
    corpus=MELD_Emotion
    max_sequence_length=512
    model_name=bilstm
    pre_trained_model=False
    frame_len=0.01
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs10ms_clean_noise123_music123
    no_epochs=100
    suffix=_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_spkrind_MELD_Emotion
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_Emotion_ConvClassif_Longformer_fs100ms_spkrind ]]; then
    corpus=MELD_Emotion
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs100ms_clean_noise123_music123
    no_epochs=100
    suffix=_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_spkrind_MELD_Emotion
    test_file=test.tsv
    results_suffix=.pkl
fi


if [[ $task_name == MELD_Emotion_ConvClassif_bilstm_fs100ms_spkrind ]]; then
    corpus=MELD_Emotion
    max_sequence_length=512
    model_name=bilstm
    pre_trained_model=False
    frame_len=0.1
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs100ms_clean_noise123_music123
    no_epochs=100
    suffix=_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_spkrind_MELD_Emotion
    test_file=test.tsv
    results_suffix=.pkl
fi


if [[ $task_name == MELD_Emotion_ConvClassif_xformer_cnnop_segpool_fs10ms_spkrind ]]; then
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    corpus=MELD_Emotion
    no_epochs=50 #100
    frame_len=0.01
    model_name=xformer_cnnop_segpool #xformersegpool
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    results_suffix=_with_uttids.pkl
    
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs10ms_clean_noise123_music123
    
    suffix=${pre_train_suffix}_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_spkrind_PoolSegments_MELD_Emotion

    test_file=test.tsv
    results_suffix=_with_uttids.pkl
    
fi

if [[ $task_name == MELD_Emotion_ConvClassif_xformer_cnnop_segpool_fs10ms_spkrind_classwts ]]; then
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    corpus=MELD_Emotion
    no_epochs=50 #100
    frame_len=0.01
    model_name=xformer_cnnop_segpool #xformersegpool
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    results_suffix=_with_uttids.pkl
    #classwts=1607,361,358,2308,6434,1002,1636
    classwts=8,38,38,6,2,13,8
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_emotion_labels_spkr_ind_fs10ms_clean_noise123_music123
    
    suffix=${pre_train_suffix}_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_spkrind_PoolSegments_MELD_Emotion_classwts

    test_file=test.tsv
    results_suffix=_with_uttids.pkl
    
fi


if [[ $task_name == MELD_Sentiment_ConvClassif_xformer_cnnop_segpool_fs10ms_spkrind ]]; then
    #data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
    corpus=MELD_Sentiment
    no_epochs=50 #100
    frame_len=0.01
    model_name=xformer_cnnop_segpool #xformersegpool
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True #False #True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    results_suffix=_with_uttids.pkl
    
    data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_Sentiment_labels_spkr_ind_fs10ms_clean_noise123_music123
    
    suffix=${pre_train_suffix}_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_spkrind_PoolSegments_MELD_Sentiment

    test_file=test.tsv
    results_suffix=_with_uttids.pkl
    
fi
   
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_silenceNone_OOSNone_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone
#results_suffix=_with_uttids.pkl
#
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_NoSilence_NoOOS_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_NoSilence_NoOOS
#results_suffix=_with_uttids.pkl
#
#
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_silence_OOS_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOS
#
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_OOS_fs10ms_clean_noise123_music123
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_OOS





for label_scheme in Exact #E IE # Exact
do
for segmentation_type in smooth #fine #smooth
do
    python daseg/bin/run_journal_jobs.py --data-dir $data_dir \
             --exp-dir ${main_exp_dir}/${corpus}_CV_${cv}_${label_scheme}LabelScheme_${segmentation_type}Segmentation${suffix}/${model_name}_${corpus}_${seed} \
             --train-mode $train_mode --frame-len $frame_len \
             --label-scheme $label_scheme --segmentation-type $segmentation_type \
             --max-sequence-length $max_sequence_length \
             --use-grid $use_grid \
             --num-gpus $num_gpus \
             --batch-size $batch_size \
             --gacc $gacc \
             --results-suffix $results_suffix \
             --concat-aug $concat_aug \
             --corpus $corpus \
             --emospotloss-wt $emospotloss_wt \
             --no-epochs $no_epochs \
             --emospot-concat $emospot_concat \
             --seed $seed \
             --label-smoothing-alpha $label_smoothing_alpha \
             --model-name $model_name \
             --pre-trained-model $pre_trained_model \
             --test-file $test_file \
             --full-speech $full_speech \
             --classwts $classwts
done
done



