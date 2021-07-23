

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
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/TrueCasing_expts_data_v3
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

if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave
    
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=25 #15
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}_Epochs_${no_epochs}

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/earning21_benchmark_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=earning21_benchmark.pkl
    fi
fi

if [[ $task_name == truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=256
    model_name=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave
    
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=50 #25 #15
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}_Epochs_${no_epochs}_${hf_model_name}

    test_file=test.tsv
    results_suffix=.pkl

    #test_file=${data_dir}/train.tsv
    #results_suffix=_train.pkl

    #test_file=${data_dir}/dev.tsv
    #results_suffix=_dev.pkl

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_4classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=${eval_dataset}.pkl
    fi
fi

if [[ $task_name == crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=256
    model_name=crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave

    pretrained_full_model_path=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_loss_wts_0.1_0.9_Epochs_25_bert-base-cased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/checkpointepoch=23.ckpt
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=25 #15
    suffix=_${task_name}_loss_wts_${loss_wts}_${hf_model_name}_GuetenFullBERTcased2Fisher

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_4classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=${eval_dataset}.pkl
    fi
fi

if [[ $task_name == fromuncased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=256
    model_name=crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave

    pretrained_full_model_path=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_loss_wts_0.5_0.5_Epochs_25_bert-base-uncased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/checkpointepoch=21.ckpt
    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=25 #15
    suffix=_${task_name}_loss_wts_${loss_wts}_${hf_model_name}_GuetenFullBERTuncased2Fisher

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_4classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=${eval_dataset}.pkl
    fi
fi


if [[ $task_name == fromCNNcased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=256
    model_name=crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave

    pretrained_full_model_path=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_loss_wts_0.5_0.5_Epochs_25_bert-base-cased_CNN/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/checkpointepoch=22.ckpt

    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=25 #15
    suffix=_${task_name}_loss_wts_${loss_wts}_${hf_model_name}_CNNBERTcased2Fisher

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_4classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=${eval_dataset}.pkl
    fi
fi

if [[ $task_name == fromCNN2500cased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix} ]]
then
    ## common_suffix can be _CombineUCandMC or _WOTurnToken_CombineUCandMC or 
    ## _CombineUCandMC_8classes or _WOTurnToken_CombineUCandMC_8classes

    no_epochs=5
    max_sequence_length=256
    model_name=crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
    hf_model_name=$hf_model_name
    data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${dataset}_true_casing_punctuation${common_suffix}/ 
    #suffix=_text_model_${task_name}
    monitor_metric_mode=max
    monitor_metric=macro_f1_ave

    pretrained_full_model_path=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_2500_subsetseed_42_loss_wts_0.5_0.5_Epochs_50_bert-base-cased_CNN/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/checkpointepoch=49.ckpt

    suffix=_text_model_${task_name}_loss_wts_${loss_wts}

    no_epochs=25 #15
    suffix=_${task_name}_loss_wts_${loss_wts}_${hf_model_name}_CNNBERTcased2Fisher

    test_file=test.tsv

    if [[ $challenge_eval == 1 ]]
    then
        data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation_WOTurnToken_CombineUCandMC_4classes/
        #data_dir=/export/c04/rpapagari/truecasing_work/data_v3/${eval_dataset}_true_casing_punctuation${common_suffix}/
        test_file=${data_dir}/test.tsv
        results_suffix=${eval_dataset}.pkl
    fi
fi


if [[ $dataset != fisher ]]; then
    suffix=${suffix}_${dataset}
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
             --loss-wts $loss_wts \
             --hf-model-name $hf_model_name \
             --pretrained-full-model-path $pretrained_full_model_path
done
done



