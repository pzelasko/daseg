
set -x

############################################################

train_mode=TE
for HF_model_name in bert-base-uncased #bert-base-cased bert-base-uncased
do
for dataset_seed in 42 43 44 45 46
do
for num_samples in 10 # 9168 1000 5000 10 50 100 250 500  # 100 500 1000 5000 9168
do
    seed=42
    loss_wts=0.5_0.5
    grid=True #False #True #False #True
    gpu=1
    dataset=fisher
    challenge_eval=0
    eval_dataset=fisher
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_${dataset_seed}
    do
        #task=fromCNNcased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix}
        #task=fromCNN2500cased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix}
        #task=crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix}
        task=fromuncased_crossdomain_truecasing_punctuation_tokenclassif_BERT${common_suffix}
        bash run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset
    done

    echo $loss_wts
done
done
done


