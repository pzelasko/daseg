

cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
task_name=${8-None}
use_grid=${9-True}
challenge_eval=${10-0}
common_suffix=${11-None}
loss_wts=${12-1_1}

if [[ $challenge_eval == 1 ]]
then
    train_mode=E
    #num_gpus=1
fi
if [[ $common_suffix == None ]]
then
    common_suffix=''
fi


corpus=TrueCasing
emospotloss_wt=-100
emospot_concat=False
no_epochs=10
max_sequence_length=512
frame_len=0.01
results_suffix=.pkl
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/TrueCasing_expts_data_v2
label_smoothing_alpha=0
model_name=longformer_text_SeqClassification
pre_trained_model=False
test_file=test.tsv
full_speech=False
monitor_metric_mode=max
monitor_metric=macro_f1
speech_pretrained_model_path=None 

########################## truecasing    #############################

if [[ $task_name == truecasing_longformer_tokenclassif${common_suffix} ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing${common_suffix}/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl    
    fi

fi

if [[ $task_name == truecasingWOpunct_longformer_tokenclassif${common_suffix} ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_WOpunct${common_suffix}/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv


    suffix=_text_model_${task_name}_NoLowerTurnToken_Debug

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_WOpunct${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi

if [[ $task_name == truecasingWOTurnToken_longformer_tokenclassif${common_suffix} ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_WOTurnToken${common_suffix}/ 
    suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    #suffix=_text_model_${task_name}_Waste
    test_file=test.tsv
    
    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_WOTurnToken${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi

if [[ $task_name == truecasingWOpunctWOTurnToken_longformer_tokenclassif${common_suffix} ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_WOpunct_WOTurnToken${common_suffix}/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_WOpunct_WOTurnToken${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi

fi


##################  punctuation   ########################

if [[ $task_name == punctuation_longformer_tokenclassif${common_suffix} ]]
then
    ## common suffix can be '' _WOCasing _WOTurnToken _WOCasingWOTurnToken
    ## _WOCasingWOTurnToken_8classes _WOCasing_8classes _WOTurnToken_8classes _8classes
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}
    test_file=test.tsv
    monitor_metric_mode=min
    monitor_metric=val_loss

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl    
    fi

fi



if [[ $task_name == punctuation_LastSubTokenLossCompute_longformer_tokenclassif${common_suffix} ]]
then
    ## common suffix can be '' _WOCasing _WOTurnToken _WOCasingWOTurnToken 
    no_epochs=5
    max_sequence_length=512
    model_name=punctuation_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}
    test_file=test.tsv
    monitor_metric_mode=min
    monitor_metric=val_loss

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl    
    fi

fi





##################   truecasing and punctuation multitasking

if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=min
    monitor_metric=val_loss
    
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=15
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}_Epochs_${no_epochs}

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi

if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=min
    monitor_metric=val_loss
    
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi

if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_1Att1ClassifLinearLayers${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_1Att1ClassifLinearLayers
    data_dir=/export/c04/rpapagari/truecasing_work/data_v2/fisher_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=min
    monitor_metric=val_loss
    
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v2/earning21_benchmark_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi



for label_scheme in Exact #E IE # Exact
do
for segmentation_type in smooth #fine #smooth
do
    python daseg/bin/run_journal_jobs_TrueCasing.py --data-dir $data_dir \
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
             --pretrained-model-path $speech_pretrained_model_path \
             --loss-wts $loss_wts
done
done


