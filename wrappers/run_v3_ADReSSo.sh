
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
task_name=${8-None}
use_grid=${9-True}
diagnosis_MMSE=${10-diagnosis} # MMSE diagnosis progression
challenge_eval=${11-0} # 1 or 0 
traindevtest_traintest=${12-TrainDevTest} # TrainTest


if [[ $challenge_eval == 1 ]]
then
    train_mode=E
    num_gpus=0
fi

corpus=ADReSSo #_CV_${cv}
emospotloss_wt=-100
emospot_concat=False
no_epochs=10
max_sequence_length=512
frame_len=0.01
results_suffix=.pkl
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/ADReSSo_IS2021_expts
label_smoothing_alpha=0
model_name=longformer_text_SeqClassification
pre_trained_model=False
test_file=test.tsv
full_speech=False
monitor_metric_mode=max
monitor_metric=accuracy
speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5


########################## convclassif with x-vector features   #############################

if [[ $task_name == longformer_text_SeqClassification ]]
then
    no_epochs=10
    max_sequence_length=512
    model_name=longformer_text_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v1_Longformer/cv_${cv} # do not have test files
    suffix=text_model
    test_file=${data_dir}/dev.tsv
fi

if [[ $task_name == longformer_text_SeqClassification_TrainTrainTest ]]
then
    no_epochs=10
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v1_Longformer_TrainTest/cv_${cv}
    suffix=_text_model_TrainTest
    #suffix=_text_model_TrainTest_0.1Warmup
    model_name=longformer_text_SeqClassification
    test_file=test.tsv
fi

if [[ $task_name == longformer_text_SeqClassification ]]
then
    no_epochs=10
    max_sequence_length=512
    model_name=longformer_text_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    suffix=_text_model_${traindevtest_traintest}
    #suffix=_text_model_${traindevtest_traintest}_0.1Warmup # warmup is set in the bin/dasg script
    test_file=test.tsv
fi

if [[ $task_name == bert_text_SeqClassification ]]
then
    no_epochs=10
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    suffix=_text_model_${traindevtest_traintest}_BERT
    #suffix=_text_model_${traindevtest_traintest}_BERT_0.1Warmup # changed in the 
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}
fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v2 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v2_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v2

    transcript_version=v2
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v3 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v3_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v3
    transcript_version=v3
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v4 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v4_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v4
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_temp

    transcript_version=v4
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v5 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v5_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v5
    transcript_version=v5
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v6 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v6_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v6
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_temp
    transcript_version=v6
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi


fi

if [[ $task_name == bert_text_SeqClassification_transcripts_v7 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v7_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v7

    #model_name=bert-base-cased
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v7_Cased

    transcript_version=v7
    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_${transcript_version}_Longformer_${traindevtest_traintest}_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi


fi

if [[ $task_name == bert_text_SeqClassificationAlignmentsSilSpkr_v1 ]]
then
    ##############################################
    ## May not be used for experiments
    #############################################
    #max_sequence_length=512
    #data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_AlignmentsSilSpkr_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    #model_name=bert-base-uncased
    #test_file=test.tsv
    #
    no_epochs=30
    ##suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSilSpkr_v1_AlternatingTokenTypeEmbed
    ##suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSilSpkr_v1_NoTokenTypeEmbed
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSilSpkr_v1_1TokenTypeEmbedForINV
fi

if [[ $task_name == bert_text_SeqClassificationAlignmentsSpkrOnlyINVPAR_v1 ]]
then
    max_sequence_length=512
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_AlignmentsSpkrOnlyINVPAR_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    # suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSpkrOnlyINVPAR ## buggy (look at "CorrectedBugs")
    
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSpkrOnlyINVPAR_CorrectedBugs
    # CorrectedBugs -- mask was not used during training, tokentype_embeddings were not given during testing
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_NoSEP
 
fi

if [[ $task_name == bert_text_SeqClassification_transcripts_AlignmentsSpkrOnlyINVPAR_v2 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_AlignmentsSpkrOnlyINVPAR_v2_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_AlignmentsSpkrOnlyINVPAR_v2_NoSEP
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_temp
fi

if [[ $task_name == bert_text_SeqClassification_transcripts_AlignmentsSpkrOnlyINVPAR_v3 ]]
then
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_AlignmentsSpkrOnlyINVPAR_v3_Longformer_${traindevtest_traintest}/cv_${cv}
    model_name=bert-base-uncased
    test_file=test.tsv
    
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_AlignmentsSpkrOnlyINVPAR_v3_NoSEP
    #suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_temp
fi


if [[ $task_name == bert_text_SeqClassification_TrainTest ]]
then
    no_epochs=10
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v1_Longformer_TrainTest/cv_${cv}
    suffix=_text_model_TrainTest_BERT
    #suffix=_text_model_TrainTest_BERT_0.1Warmup # changed in the script
    model_name=bert-base-uncased
    test_file=test.tsv
fi

if [[ $task_name == bert_text_SeqClassification_Alignments_v1 ]]
then
    #########################################
    ## the input transcripts are csv paths with no alignments
    ##########################################

    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_Alignments_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_Alignments_v1
    model_name=bert-base-uncased
    test_file=test.tsv
fi

if [[ $task_name == bert_text_SeqClassification_AlignmentsNoSil_v1 ]]
then
    #########################################
    ## the input transcripts are csv paths with no alignments
    ##########################################

    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_AlignmentsNoSil_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    no_epochs=30
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_AlignmentsNoSil_v1
    model_name=bert-base-uncased
    test_file=test.tsv
fi


if [[ $task_name == bert_text_SeqClassification_transcripts_v3_SimilarDynamicRange ]]
then
    no_epochs=30
    max_sequence_length=512
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v3_Longformer_${traindevtest_traintest}_SimilarDynamicRange/cv_${cv}
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v3_SimilarDynamicRange

    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v3_Longformer_${traindevtest_traintest}_SimilarDynamicRangeV2/cv_${cv}
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v3_SimilarDynamicRangeV2

    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_text_v3_Longformer_${traindevtest_traintest}_SimilarDynamicRangeV3/cv_${cv}
    suffix=_text_model_${traindevtest_traintest}_BERT_Epochs_${no_epochs}_transcripts_v3_SimilarDynamicRangeV3

    model_name=bert-base-uncased
    test_file=test.tsv
    
fi



########################################
if [[ $task_name == MultiModal_fs10ms_BERT ]]
then
    max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
    no_epochs=100
    frame_len=0.01
    model_name=TransformerMultiModalSeqClassification ## for multimodal models, x-vector pretrained model is used always
    test_file=test.tsv
    pre_trained_model=True
    
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_Multimodal_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    #suffix=_text_model_${traindevtest_traintest}_MultiModal_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}
    suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers
    ##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_1CrossAttLayers
    ##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_3CrossAttLayers
    ##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_3CrossAttLayers_DropoutOnConcatVec
fi

if [[ $task_name == MultiModal_fs100ms_BERT ]]
then
    max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
    no_epochs=100
    frame_len=0.1
    model_name=TransformerMultiModalSeqClassification ## for multimodal models, x-vector pretrained model is used always
    test_file=test.tsv
    pre_trained_model=True
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_Multimodal_fs100ms_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers
    ##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_speechatt
fi

#######################################
if [[ $task_name == MultiModalMultiLoss_fs100ms_BERT ]]
then
    max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
    no_epochs=100
    frame_len=0.1
    model_name=TransformerMultiModalMultiLossSeqClassification ## for multimodal models, x-vector pretrained model is used always
    test_file=test.tsv
    pre_trained_model=True
    #data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_MultimodalMultiloss_fs100ms_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_Multimodal_fs100ms_v1_Longformer_${traindevtest_traintest}/cv_${cv}
    #suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_EqualWts # wts are changed in the bin/dasg script
    suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_2_1_Wts
    #suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_10_1_Wts

fi

if [[ $task_name == MultiModalMultiLoss_fs100ms_BERT_transcripts_v7 ]]
then
    max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
    no_epochs=100
    frame_len=0.1
    model_name=TransformerMultiModalMultiLossSeqClassification ## for multimodal models, x-vector pretrained model is used always
    test_file=test.tsv
    pre_trained_model=True

    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_Multimodal_fs100ms_v7_Longformer_${traindevtest_traintest}/cv_${cv}
    suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_2_1_Wts_transcripts_v7
    suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_0CrossAttLayers_2_1_Wts_transcripts_v7_warmp0.1

fi



########################################
if [[ $task_name == longformer_fs10ms_cleanspeech_seqclassif ]] 
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}/cv_${cv}
    no_epochs=20
    frame_len=0.01
    max_sequence_length=2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=longformer_speech_SeqClassification

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs10ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi

   
if [[ $task_name == resnet_aug_fs10ms_cleanspeech_seqclassif ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}/cv_${cv}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs10ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi

if [[ $task_name == resnet_aug_fs10ms_augspeech_seqclassif ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_clean_noise123_music123/cv_${cv}
    #suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug_maxlen_${max_sequence_length}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs10ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi

if [[ $task_name == resnet_aug_fs10ms_cleanspeech_seqclassif_spkrPAR ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_spkrPAR/cv_${cv}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_spkrPAR
    #suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_spkrPAR_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs10ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi

if [[ $task_name == resnet_aug_fs100ms_cleanspeech_seqclassif ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi

if [[ $task_name == resnet_aug_fs100ms_augspeech_seqclassif ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_clean_noise123_music123/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi

if [[ $task_name == resnet_aug_fs100ms_cleanspeech_seqclassif_PARaudio ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_PARaudio/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_PARaudio
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi

if [[ $task_name == resnet_aug_fs100ms_augspeech_seqclassif_PARaudio ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_PARaudio_clean_noise123_music123/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}_PARaudio
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi


#############################  resnet clean -- speech ############
if [[ $task_name == resnet_clean_fs100ms_cleanspeech_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    test_file=test.tsv
    model_name=resnet_SeqClassification

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi

fi
if [[ $task_name == resnet_clean_fs100ms_augspeech_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_clean_noise123_music123/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    test_file=test.tsv
    model_name=resnet_SeqClassification

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_challenge_test/cv_1/
        test_file=${data_dir}/challenge_test.tsv
        results_suffix=challenge_test.pkl
    fi
fi
    
if [[ $task_name == resnet_clean_fs100ms_cleanspeech_seqclassif_PARaudio ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_PARaudio/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_PARaudio
    test_file=test.tsv
    model_name=resnet_SeqClassification
fi
if [[ $task_name == resnet_clean_fs100ms_augspeech_seqclassif_PARaudio ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_fs100ms_PARaudio_clean_noise123_music123/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}_PARaudio
    test_file=test.tsv
    model_name=resnet_SeqClassification
fi


############################## bilstm ###
if [[ $task_name == bilstm_fs100ms_cleanspeech_seqclassif_vggish ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_${diagnosis_MMSE}_cv10_8k_preprocessed_${traindevtest_traintest}_vggish/cv_${cv}
    no_epochs=50
    frame_len=1
    max_sequence_length=300 # 4000  #2048
    suffix=_BiLSTM_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_vggish
    suffix=_BiLSTM_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_vggish_batchnormiplayer
    no_epochs=200
    suffix=_BiLSTM_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_vggish_batchnormiplayer_meanpooling
    suffix=_BiLSTM_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_${traindevtest_traintest}_maxlen_${max_sequence_length}_Epochs_${no_epochs}_vggish_batchnormiplayer_meanpooling_2BiLSTMlayers
    test_file=test.tsv
    model_name=bilstm_SeqClassification
fi

if [[ $diagnosis_MMSE == MMSE ]]
then
    monitor_metric_mode=min
    monitor_metric=val_loss
    suffix=${suffix}_${diagnosis_MMSE}
    #suffix=${suffix}_${diagnosis_MMSE}_L1Loss # changed in the code itself
fi

if [[ $diagnosis_MMSE == progression ]]
then
    suffix=${suffix}_${diagnosis_MMSE}
    no_epochs=5 # data is very less so running for very few epochs 
fi


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
             --monitor-metric $monitor_metric \
             --monitor-metric-mode $monitor_metric_mode \
             --pretrained-model-path $speech_pretrained_model_path
done
done


