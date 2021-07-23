
set -x



source /home/rpapagari/.bashrc

source activate daseg_v2

train_mode=TE
for HF_model_name in bert-base-uncased #bert-base-cased bert-base-uncased
do
for seed in 42 #43 44 # 42
do
for loss_wts in  0.1_0.9 0.9_0.1 0.75_0.25 # 1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
# 0.95_0.05
do
    grid=True #False #True
    gpu=1
    dataset=fisher #guetenburg
    #dataset=fisher
    challenge_eval=0
    eval_dataset=fisher

    for common_suffix in _WOTurnToken_CombineUCandMC_8classes_100_subsetseed_42 _WOTurnToken_CombineUCandMC_8classes_100_subsetseed_43 _WOTurnToken_CombineUCandMC_8classes_100_subsetseed_44 _WOTurnToken_CombineUCandMC_8classes_500_subsetseed_42 _WOTurnToken_CombineUCandMC_8classes_500_subsetseed_43 _WOTurnToken_CombineUCandMC_8classes_500_subsetseed_44 _WOTurnToken_CombineUCandMC_8classes_1000_subsetseed_42 _WOTurnToken_CombineUCandMC_8classes_1000_subsetseed_43 _WOTurnToken_CombineUCandMC_8classes_1000_subsetseed_44 _WOTurnToken_CombineUCandMC_8classes_5000_subsetseed_42

    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix}
        bash run_TrueCasing_data_v3_Sonal.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset
    done

    echo $loss_wts
done
done
done

exit

