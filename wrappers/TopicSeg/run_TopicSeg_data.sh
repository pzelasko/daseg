

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
dataset=${13-fisher}
hf_model_name=${14-bert-base-cased}
eval_dataset=${15-None}
gpu_ind=${16--1}

if [[ $challenge_eval == 1 ]]
then
    train_mode=E
    #num_gpus=2
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
main_exp_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_expts/  #/export/c02/rpapagari/daseg_erc/daseg/TrueCasing_expts_data_v3
#main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/TrueCasing_expts_data_v3_exptwts
label_smoothing_alpha=0
model_name=longformer_text_SeqClassification
pre_trained_model=False
test_file=test.tsv
full_speech=False
monitor_metric_mode=max
monitor_metric=macro_f1
speech_pretrained_model_path=None 
pretrained_full_model_path=None

##################   truecasing and punctuation multitasking


if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix} ]]
then
    no_epochs=50
    max_sequence_length=256
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    #data_dir=/home/rpappagari/daseg/daseg/topic_seg_utils/temp
    #data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_cv10/cv_${cv}/  ## used first but then folds in CV are formed with shuffle so results could be different with it
    
    #data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${dataset}cv10/cv_${cv}/

    data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${dataset}cv10/cv_${cv}/

    monitor_metric_mode=max
    monitor_metric=macro_f1_ave

    #no_epochs=50 #25 #15
    if [[ $hf_model_name == *"/"* ]]; then
        hf_model_name_temp=`echo $hf_model_name | cut -d '/' -f2`
    else
        hf_model_name_temp=$hf_model_name
    fi
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}_Epochs_${no_epochs}_${hf_model_name_temp}

    test_file=test.tsv
    results_suffix=.pkl

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${eval_dataset}cv10/
        test_file=${data_dir}/all.tsv  
        results_suffix=_${eval_dataset}.pkl
    fi
fi

if [[ $task_name == topicseg_Morethan2TasksArch_SeqClassif_BERT${common_suffix} ]]
then
    no_epochs=25 #50
    max_sequence_length=256
    model_name=topicseg_Morethan2TasksArch_SeqClassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${dataset}cv10/cv_${cv}/

    monitor_metric_mode=max
    ## using "macro_f1_op0" as met
    monitor_metric=macro_f1_op0 #macro_f1_ave

    #no_epochs=50 #25 #15
    if [[ $hf_model_name == *"/"* ]]; then
        hf_model_name_temp=`echo $hf_model_name | cut -d '/' -f2`
    else
        hf_model_name_temp=$hf_model_name
    fi
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}_Epochs_${no_epochs}_${hf_model_name_temp}

    test_file=test.tsv
    results_suffix=.pkl

    if [[ $challenge_eval == 1 ]]; then
        data_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${eval_dataset}cv10/
        test_file=${data_dir}/all.tsv  
        results_suffix=_${eval_dataset}.pkl
    fi
fi



if [[ $dataset != fisher ]]; then
    suffix=${suffix}_${dataset}
fi


expt_dir=${main_exp_dir}/${corpus}_CV_${cv}_${label_scheme}LabelScheme_${segmentation_type}Segmentation${suffix}/${model_name}_${corpus}_${seed}
mkdir -p $expt_dir

for label_scheme in Exact #E IE # Exact
do
for segmentation_type in smooth #fine #smooth
do
    python daseg/bin/run_journal_jobs_TopicSeg.py --data-dir $data_dir \
             --exp-dir ${expt_dir} \
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
             --loss-wts $loss_wts \
             --hf-model-name $hf_model_name \
             --pretrained-full-model-path $pretrained_full_model_path \
             --gpu-ind $gpu_ind 2>&1 | tee -a ${expt_dir}/log.txt
done
done



