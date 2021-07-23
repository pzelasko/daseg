
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
use_grid=${8-True}


corpus=IEMOCAP_v2 #_CV_${cv}
emospotloss_wt=-100
emospot_concat=False
no_epochs=50
max_sequence_length=512
frame_len=0.1
results_suffix=.pkl
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/journal_v2
label_smoothing_alpha=0
model_name=longformer
pre_trained_model=False
test_file=test.tsv
full_speech=False

######################### convclassif with x-vector features   #############################
#xvector_model_id=SoTAEnglishXVector_IsolatedUttTraining_run1
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/xvectorfeat_data_dirs/${xvector_model_id}/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#no_epochs=100
#frame_len=0.08
#suffix=${xvector_model_id}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}
#

############ re-run to check if we get same result because code-base is reorganized a bit and results are not same after that
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
#no_epochs=100
#frame_len=0.1
#suffix=${xvector_model_id}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_temp

######################## convclassif    #############################
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
#no_epochs=100
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps


#model_name=bilstm
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}

####################  convclassif with xformer and fs=10ms  ######################
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
no_epochs=100
frame_len=0.01
model_name=xformer
max_sequence_length=2048
xvector_model_id=_SoTAEnglishXVector
pre_trained_model=True #False #True
pre_train_suffix=${xvector_model_id}${pre_trained_model}
results_suffix=_with_uttids.pkl

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOS_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOS
#results_suffix=_with_uttids.pkl
#results_suffix=_all_with_uttids.pkl
#test_file=${data_dir}/all.tsv

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOS_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOS_spkrind
#results_suffix=_with_uttids.pkl

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOS_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_with_uttids_InferenceWOspkrind.pkl

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOSNone_spkrind
##full_speech=True
#results_suffix=_with_uttids.pkl

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_spkrind
##full_speech=True
#results_suffix=_with_uttids.pkl

################## only for evaluation
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrM_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_with_uttids_SpkrM.pkl
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrF_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_with_uttids_SpkrF.pkl
##############

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone
###full_speech=True
#results_suffix=_with_uttids.pkl

################## only for evaluation
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrM_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_with_uttids_SpkrM.pkl
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_SpkrF_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_with_uttids_SpkrF.pkl
################

#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind
#results_suffix=_with_uttids.pkl

#data_dir=/export/b17/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_NeutralNone_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_NeutralNone
#results_suffix=_with_uttids.pkl


#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOSNone_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOSNone
##full_speech=True
#results_suffix=_with_uttids.pkl


#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_NoSilence_OOS_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_NoSilence_OOS


#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_NoSilence_AllClasses_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_NoSilence_AllClasses


#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_ASHNF_ExciteMapHap_fs10ms_UttEval.pkl


#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_silenceNone_OOSNone_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone
#results_suffix=_with_uttids.pkl

#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_NoSilence_NoOOS_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_NoSilence_NoOOS
#results_suffix=_with_uttids.pkl


#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_silence_OOS_fs10ms_clean_noise123_music123/
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOS
#
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_OOS_fs10ms_clean_noise123_music123
#corpus=SWBD_v2
#suffix=${pre_train_suffix}_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_${model_name}_maxseqlen_${max_sequence_length}_OOS


######################  convclassif with xformersegpool and fs=10ms  ######################
##data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#no_epochs=100
#frame_len=0.01
#model_name=xformersegpool #xformer_cnnop_segpool #xformersegpool
#max_sequence_length=2048
#xvector_model_id=_SoTAEnglishXVector
#pre_trained_model=True #False #True
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#results_suffix=_with_uttids.pkl
#
##data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silence_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
##suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments
##results_suffix=_with_uttids.pkl
#
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_ExciteMapHap_smoothed_overlap_silenceNone_OOSNone_spkrind_fs10ms_clean_noise123_music123/cv_${cv}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_NoiseAug_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_CP_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}_smoothed_overlap_silenceNone_OOSNone_spkrind_PoolSegments
#results_suffix=_with_uttids.pkl



#####################  convclassif with ResNet34 and fs=10ms  ######################
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#no_epochs=100
#frame_len=0.01
#model_name=ResNet
#max_sequence_length=2048
#xvector_model_id=_SoTAEnglishXVector
#pre_trained_model=True #False #True
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}
#

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
#
#
################### SWBD convclassif with smoothed_overlap_silence_OOS    #############################
#corpus=SWBD_v2
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_smoothed_overlap_silence_OOS_clean_noise123_music123
#no_epochs=100
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_smoothed_overlap_silence_OOS
#results_suffix=.pkl

#corpus=SWBD_v2
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_ERC_OOS_clean_noise123_music123
#no_epochs=100
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_OOS
#results_suffix=.pkl



######################   uttclassif   #############################
###data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms/cv_${cv}
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms_clean_noise123_music123/cv_${cv}
#no_epochs=100
###suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
##suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps
##label_smoothing_alpha=0.05 #0.1 #0.05
##suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_warmup200steps_labelsmoothing${label_smoothing_alpha}
#results_suffix=.pkl
#
#model_name=bilstm
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}
#
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
#results_suffix=_conversations.pkl

#######################  SWBD utt classification
#corpus=SWBD_v2
#no_epochs=50
#max_sequence_length=512
#model_name=bilstm
#pre_trained_model=False
#results_suffix=_with_uttids.pkl
#frame_len=0.01
#
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_isolated_utts_fs10ms/
#suffix=_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
#results_suffix=_with_uttids.pkl

corpus=SWBD_v2 #_CV_${cv}
no_epochs=50
max_sequence_length=512
model_name=longformer
pre_trained_model=False
results_suffix=_with_uttids.pkl
frame_len=0.01
data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_isolated_utts_fs10ms/
suffix=_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
results_suffix=_with_uttids.pkl

#corpus=SWBD_v2 #_CV_${cv}
#no_epochs=50
#max_sequence_length=2048
#model_name=ResNet
#pre_trained_model=True
#xvector_model_id=_SoTAEnglishXVector
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#results_suffix=_with_uttids.pkl
#frame_len=0.01
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_isolated_utts_fs10ms/
#suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxseqlen_${max_sequence_length}
#results_suffix=_with_uttids.pkl
#
#corpus=SWBD_v2 #_CV_${cv}
#no_epochs=50
#max_sequence_length=2048
#model_name=xformer
#pre_trained_model=True
#xvector_model_id=_SoTAEnglishXVector
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#results_suffix=_with_uttids.pkl
#frame_len=0.01
#data_dir=/export/b15/rpapagari/Tianzi_work/SWBD/data_v2_isolated_utts_fs10ms/
#suffix=${pre_train_suffix}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_maxseqlen_${max_sequence_length}
#results_suffix=_with_uttids.pkl


#####################  uttclassif with ResNet34 and fs=10ms  ######################
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#no_epochs=100
#frame_len=0.01
#model_name=ResNet
#max_sequence_length=2048
#xvector_model_id=_SoTAEnglishXVector
#pre_trained_model=True #False #True
#pre_train_suffix=${xvector_model_id}${pre_trained_model}
#suffix=${pre_train_suffix}_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_${model_name}_maxseqlen_${max_sequence_length}
#
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_fs10ms_clean_noise123_music123/cv_${cv}
#results_suffix=_conversations.pkl

################## emospot  #############################
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ASHNF_ExciteMapHap_fs100ms_clean_noise123_music123/cv_${cv}/,/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
#emospotloss_wt=3.0
#emospot_concat=True
#no_epochs=100
#suffix=_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len${frame_len}_${emospotloss_wt}EmoSpot_EmoSpotConcat_${emospot_concat}_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}
#results_suffix=.pkl
#
#data_dir=/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_v2_ERC_NoInfreqEmo_ExciteMapHap_clean_noise123_music123/cv_${cv}
#results_suffix=_conversations.pkl

#--exp-dir ${main_exp_dir}/${corpus}_CV_${cv}_${label_scheme}LabelScheme_${segmentation_type}Segmentation${suffix} \

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
             --full-speech $full_speech
done
done


