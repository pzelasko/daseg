
cv=$1
train_mode=$2
num_gpus=${3-1}
batch_size=${4-6}
gacc=${5-1}
concat_aug=${6--1}
seed=${7-42}
task_name=${8-None}
use_grid=${9-True}


if [[ $challenge_eval == 1 ]]
then
    train_mode=E
    num_gpus=0
fi

corpus=TrueCasing
emospotloss_wt=-100
emospot_concat=False
no_epochs=10
max_sequence_length=512
frame_len=0.01
results_suffix=.pkl
main_exp_dir=/export/c02/rpapagari/daseg_erc/daseg/TrueCasing_expts
label_smoothing_alpha=0
model_name=longformer_text_SeqClassification
pre_trained_model=False
test_file=test.tsv
full_speech=False
monitor_metric_mode=max
monitor_metric=macro_f1
speech_pretrained_model_path=None 


########################## convclassif with x-vector features   #############################

if [[ $task_name == truecasing_longformer_tokenclassif ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data/fisher_true_casing/ 
    #suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv

    data_dir=/export/c04/rpapagari/truecasing_work/data/earning21_benchmark/
    test_file=${data_dir}/test.tsv
    results_suffix=earning21_benchmark.pkl    

fi

if [[ $task_name == truecasingWOpunct_longformer_tokenclassif ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data/fisher_true_casing_WOpunct/ 
    suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv

    data_dir=/export/c04/rpapagari/truecasing_work/data/earning21_benchmark_WOpunct/
    test_file=${data_dir}/test.tsv
    results_suffix=earning21_benchmark.pkl
fi

if [[ $task_name == truecasingWOTurnToken_longformer_tokenclassif ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data/fisher_true_casing_WOTurnToken/ 
    suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    #suffix=_text_model_${task_name}_Waste
    test_file=test.tsv
    
    data_dir=/export/c04/rpapagari/truecasing_work/data/earning21_benchmark_WOTurnToken/
    test_file=${data_dir}/test.tsv
    results_suffix=earning21_benchmark.pkl
fi

if [[ $task_name == truecasingWOpunctWOTurnToken_longformer_tokenclassif ]]
then
    no_epochs=5
    max_sequence_length=512
    model_name=truecasing_longformer_tokenclassif
    data_dir=/export/c04/rpapagari/truecasing_work/data/fisher_true_casing_WOpunct_WOTurnToken/ 
    suffix=_text_model_${task_name}
    suffix=_text_model_${task_name}_NoLowerTurnToken
    test_file=test.tsv

    data_dir=/export/c04/rpapagari/truecasing_work/data/earning21_benchmark_WOpunct_WOTurnToken/
    test_file=${data_dir}/test.tsv
    results_suffix=earning21_benchmark.pkl

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
             --pretrained-model-path $speech_pretrained_model_path
done
done


