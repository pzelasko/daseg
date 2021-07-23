
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
task_name=${8-None}
use_grid=${9-True}



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
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer/cv_${cv}
#suffix=text_model
#test_file=${data_dir}/dev.tsv
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer_TrainDevTest/cv_${cv}
##suffix=_text_model_TrainDevTest
#suffix=_text_model_TrainDevTest_0.1Warmup
#model_name=longformer_text_SeqClassification
#test_file=test.tsv

#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer_TrainDevTest/cv_${cv}
#suffix=_text_model_TrainDevTest_BERT
##suffix=_text_model_TrainDevTest_BERT_0.1Warmup # changed in the 
#model_name=bert-base-uncased
#test_file=test.tsv

#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer_TrainDevTest/cv_${cv}
#no_epochs=30
#suffix=_text_model_TrainDevTest_BERT_Epochs_${no_epochs}
#model_name=bert-base-uncased
#test_file=test.tsv
#
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_Alignments_v1_Longformer_TrainDevTest/cv_${cv}
#no_epochs=30
#suffix=_text_model_TrainDevTest_BERT_Epochs_${no_epochs}_Alignments_v1
#model_name=bert-base-uncased
#test_file=test.tsv

#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_AlignmentsNoSil_v1_Longformer_TrainDevTest/cv_${cv}
#no_epochs=30
#suffix=_text_model_TrainDevTest_BERT_Epochs_${no_epochs}_AlignmentsNoSil_v1
#model_name=bert-base-uncased
#test_file=test.tsv


#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer_TrainTest/cv_${cv}
##suffix=_text_model_TrainTest_BERT
#suffix=_text_model_TrainTest_BERT_0.1Warmup # changed in the script
#model_name=bert-base-uncased
#test_file=test.tsv
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v1_Longformer_TrainTest/cv_${cv}
##suffix=_text_model_TrainTest
#suffix=_text_model_TrainTest_0.1Warmup
#model_name=longformer_text_SeqClassification
#test_file=test.tsv


########################################
#max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
#no_epochs=100
#frame_len=0.01
#model_name=TransformerMultiModalSeqClassification ## for multimodal models, x-vector pretrained model is used always
#test_file=test.tsv
#pre_trained_model=True
#
##data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_Multimodal_v1_Longformer_TrainDevTest/cv_${cv}
##suffix=_text_model_TrainDevTest_MultiModal_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}
##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers
###suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_1CrossAttLayers
###suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_3CrossAttLayers
###suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_3CrossAttLayers_DropoutOnConcatVec
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_Multimodal_fs100ms_v1_Longformer_TrainDevTest/cv_${cv}
#frame_len=0.1
#suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers
##suffix=_frame_len${frame_len}_MultiModal_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_speechatt

#######################################
#max_sequence_length=1000 # for speech part, for text part the default one for BERT 512 is used
#no_epochs=100
#frame_len=0.1
#model_name=TransformerMultiModalMultiLossSeqClassification ## for multimodal models, x-vector pretrained model is used always
#test_file=test.tsv
#pre_trained_model=True
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_MultimodalMultiloss_fs100ms_v1_Longformer_TrainDevTest/cv_${cv}
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_Multimodal_fs100ms_v1_Longformer_TrainDevTest/cv_${cv}
#frame_len=0.1
#suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_EqualWts
#suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_2_1_Wts
#suffix=_frame_len${frame_len}_MultiModalMultiLoss_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxlenspeech_${max_sequence_length}_Epochs_${no_epochs}_2CrossAttLayers_10_1_Wts



########################################
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest/cv_${cv}
#no_epochs=20
#max_sequence_length=2048
#xvector_model_id=_SoTAEnglishXVector
#pre_trained_model=True
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
#test_file=test.tsv
#model_name=longformer_speech_SeqClassification
#
#data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest/cv_${cv}
#no_epochs=20
#max_sequence_length=4000 # 4000  #2048
#xvector_model_id=_SoTAEnglishXVector
#pre_trained_model=True
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
##suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
#suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}
#test_file=test.tsv
#model_name=resnet_SeqClassification

if [[ $task_name == resnet_aug_fs10ms_cleanspeech_seqclassif ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest/cv_${cv}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi

if [[ $task_name == resnet_aug_fs10ms_augspeech_seqclassif ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_clean_noise123_music123/cv_${cv}
    #suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_${max_sequence_length}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi

if [[ $task_name == resnet_aug_fs10ms_cleanspeech_seqclassif_spkrPAR ]]
then
    no_epochs=50
    frame_len=0.01
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_spkrPAR/cv_${cv}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}_spkrPAR
    #suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}_spkrPAR_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi

if [[ $task_name == resnet_aug_fs100ms_cleanspeech_seqclassif ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_fs100ms/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi

if [[ $task_name == resnet_aug_fs100ms_augspeech_seqclassif ]]
then
    no_epochs=50
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest
    test_file=test.tsv
    model_name=resnet_SeqClassification
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_fs100ms_clean_noise123_music123/cv_${cv}
    frame_len=0.1
    no_epochs=50
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5
fi


#############################  resnet clean ############
if [[ $task_name == resnet_clean_fs100ms_cleanspeech_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_fs100ms/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    test_file=test.tsv
    model_name=resnet_SeqClassification
fi
if [[ $task_name == resnet_clean_fs100ms_augspeech_seqclassif ]]
then
    data_dir=/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_8k_preprocessed_TrainDevTest_fs100ms_clean_noise123_music123/cv_${cv}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=8000 # 4000  #2048
    xvector_model_id=_SoTAEnglishXVectorClean
    speech_pretrained_model_path=/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_clean_xvector.h5
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_${max_sequence_length}_Epochs_${no_epochs}
    test_file=test.tsv
    model_name=resnet_SeqClassification
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


