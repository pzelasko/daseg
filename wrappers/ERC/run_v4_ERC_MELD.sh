
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
use_grid=${8-True}
task_name=${9-ConvClassif_Longformer}
emotion_sentiment=${10-Emotion}
noise_aug=${11-False}
main_exp_dir=${12-/export/c02/rpapagari/daseg_erc/daseg/ERC_MELD_expts/}

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
#main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/ERC_MELD_expts/
label_smoothing_alpha=0
full_speech=False
classwts=1

##########
# you should have tasks for longformer, bilstm, ResNet, xformer, 2-stage training of ResNet and longformer
# uttclassif, conversations
# IEMOCAP, SWBD
# 

#corpus, max_sequence_length, model_name, pre_trained_model, frame_len, data_dir, no_epochs, suffix, test_file, results_suffix

if [[ $task_name == MELD_${emotion_sentiment}_UttClassif_bilstm_fs100ms ]]
then
    corpus=MELD_${emotion_sentiment}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=512 # 4000  #2048
    test_file=test.tsv
    model_name=bilstm
    pre_trained_model=False
 
    suffix=_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_MELD_${emotion_sentiment}
    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/MELD_${emotion_sentiment}_8k_fs100ms_allutts_v2_clean_noise123_music123/
        suffix=${suffix}_NoiseAug
    else
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/MELD_${emotion_sentiment}_8k_fs100ms_allutts_v2/
    fi
   
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_UttClassif_ResNet_fs10ms ]]
then
    corpus=MELD_${emotion_sentiment}
    no_epochs=50
    frame_len=0.01
    max_sequence_length=512 # 4000  #2048
    test_file=test.tsv
    model_name=ResNet
    pre_trained_model=False

    suffix=_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_MELD_${emotion_sentiment}
    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/MELD_${emotion_sentiment}_8k_fs10ms_allutts_v2_clean_noise123_music123/
        suffix=${suffix}_NoiseAug
    else
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/MELD_${emotion_sentiment}_8k_fs10ms_allutts_v2/
    fi
    
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_UttClassif_longformer_fs100ms ]]
then
    corpus=MELD_${emotion_sentiment}
    no_epochs=50
    frame_len=0.1
    max_sequence_length=512 # 4000  #2048
    test_file=test.tsv
    model_name=longformer
    pre_trained_model=False

    suffix=_${model_name}_frame_len${frame_len}_UttClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_Epochs_${no_epochs}_MELD_${emotion_sentiment}
    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_${emotion_sentiment}_8k_fs100ms_allutts_v2_clean_noise123_music123/
        suffix=${suffix}_NoiseAug
    else
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_dirs/meld_${emotion_sentiment}_8k_fs100ms_allutts_v2/
    fi
   
    results_suffix=.pkl
fi


if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_Longformer_fs100ms_spkrind ]]; then
    echo getting error somehow please check it
    exit

    corpus=MELD_${emotion_sentiment}
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    no_epochs=100
    suffix=_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_spkrind_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_spkr_ind_fs100ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_Longformer_fs100ms ]]; then
    corpus=MELD_${emotion_sentiment}
    max_sequence_length=512
    model_name=longformer
    pre_trained_model=False
    frame_len=0.1
    no_epochs=100
    suffix=_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs100ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi


if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_bilstm_fs100ms ]]; then
    corpus=MELD_${emotion_sentiment}
    max_sequence_length=512
    model_name=bilstm
    pre_trained_model=False
    frame_len=0.1
    no_epochs=100
    suffix=_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs100ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_ResNet_fs10ms ]]; then
    corpus=MELD_${emotion_sentiment}
    max_sequence_length=512
    model_name=ResNet
    pre_trained_model=False
    frame_len=0.01
    no_epochs=100
    suffix=_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs10ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi



if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_xformer_fs10ms_spkrind ]]; then

    corpus=MELD_${emotion_sentiment}
    max_sequence_length=2048
    model_name=xformer
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    frame_len=0.01
    no_epochs=100
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_spkrind_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_spkr_ind_fs10ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_xformer_fs10ms ]]; then

    corpus=MELD_${emotion_sentiment}
    max_sequence_length=2048
    model_name=xformer
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    frame_len=0.01
    no_epochs=100
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs10ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_XFormerAddSpeakerEmb_fs10ms ]]; then
    corpus=MELD_${emotion_sentiment}
    max_sequence_length=2048
    model_name=XFormerAddSpeakerEmb
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    frame_len=0.01
    no_epochs=100
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs10ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
fi

if [[ $task_name == MELD_${emotion_sentiment}_ConvClassif_XFormerConcatSpeakerEmb_fs10ms ]]; then
    corpus=MELD_${emotion_sentiment}
    max_sequence_length=2048
    model_name=XFormerConcatSpeakerEmb
    xvector_model_id=_SoTAEnglishXVector
    pre_trained_model=True
    pre_train_suffix=${xvector_model_id}${pre_trained_model}
    frame_len=0.01
    no_epochs=100
    suffix=${pre_train_suffix}_${model_name}_frame_len${frame_len}_ConvClassif_bs${batch_size}_gacc${gacc}_concat_aug_${concat_aug}_MELD_${emotion_sentiment}

    if [[ $noise_aug == True ]]; then
        data_dir=/export/b15/rpapagari/Tianzi_work/MELD_audio/data_conv_dirs/data_ERC_all_${emotion_sentiment}_labels_fs10ms_clean_noise123_music123
        suffix=${suffix}_NoiseAug
    else
        exit
    fi
    test_file=test.tsv
    results_suffix=.pkl
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
             --classwts $classwts
done
done



