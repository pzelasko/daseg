
set -x

############################################################

train_mode=TE
for HF_model_name in bert-base-cased #bert-base-cased bert-base-uncased
do
for loss_wts in 1_0 # 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9
do
    seed=42
    grid=False #True ## Always False on voicelab server
    gpu=1 # 1 if gpu is used otherwise 0
    dataset=fisher #guetenburg fisher CNN
    challenge_eval=0
    gpu_ind=6 #-1
    eval_dataset=fisher
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes

    ###### Possible options:  _WithFisherMultiTaskCasingPred
        # _WithFisherMultiTaskPunctPred
        # _WithFisherCasingPred
        # _WithFisherPunctPred
    
        # _WithPunct_WOTurnToken_CombineUCandMC_8classes
        # _WithPunct_WithSepToken_CombineUCandMC_8classes
        # _WithCasing_WOTurnToken_CombineUCandMC_8classes 
        # _WithCasing_WithSepToken_CombineUCandMC_8classes
        # _WithSepToken_CombineUCandMC_8classes 
    
        #_WithCasing_WOTurnToken_CombineUCandMC_8classes
        #_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_${dataset_seed}
    
        #_WOTurnToken_CombineUCandMC_8classes
        #_WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
    
        # _WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
        # _WithCasing_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
    
        # _WithCasing_WOTurnToken_CombineUCandMC_8classes
        # _WithPunct_WOTurnToken_CombineUCandMC_8classes
    
        #_WOTurnToken_CombineUCandMC_8classes
        #_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
        #_WOTurnToken_CombineUCandMC_4classes
        #_WOTurnToken_CombineUCandMC_4classes_9168TrainSamples

    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix}
        bash wrappers/run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset $gpu_ind
    done
    echo $loss_wts
done
done

exit

#######################  wrapper to run on subsets of data ###################

train_mode=TE
for HF_model_name in bert-base-uncased #bert-base-cased bert-base-uncased
do
for loss_wts in  1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9 
do
    seed=42
    grid=True #False #True
    gpu=1
    dataset=fisher #guetenburg
    challenge_eval=0
    gpu_ind=6 #-1
    eval_dataset=fisher
    for dataset_seed in 42 43 44 45 46
    do
    for num_samples in 10 #50 100 250 500 # 5000 9168
    do
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_${dataset_seed}
    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix}
        bash wrappers/run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset $gpu_ind
    done
    done
    done
    echo $loss_wts
done
done

