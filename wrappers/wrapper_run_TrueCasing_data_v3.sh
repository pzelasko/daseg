
set -x

############################################################

#/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithPunct_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42//summary_results.txt_label_block_name0


#train_mode=E
#for HF_model_name in bert-base-cased #bert-base-cased bert-base-uncased
#do
#for loss_wts in 1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9
#
##1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9
##1_0.1 1_0.25 1_0.5 1_0.75 1_1 0.1_1 0.25_1 0.5_1 0.75_1 
#
## 0.75_0.25 # 1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
#do
#    seed=42
#    grid=True #False #True
#    gpu=1
#    dataset=CNN #fisher #guetenburg
#    challenge_eval=1 #0
#    eval_dataset=fisher
#    for common_suffix in _WOTurnToken_CombineUCandMC_8classes_2500_subsetseed_42
#
#    # data_v3/CNN_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes
#    # _WithFisherMultiTaskCasingPred
#    # _WithFisherMultiTaskPunctPred
#    # _WithFisherCasingPred
#    # _WithFisherPunctPred
#
#    # _WithPunct_WOTurnToken_CombineUCandMC_8classes
#    # _WithPunct_WithSepToken_CombineUCandMC_8classes
#    # _WithCasing_WOTurnToken_CombineUCandMC_8classes 
#    # _WithCasing_WithSepToken_CombineUCandMC_8classes
#    # _WithSepToken_CombineUCandMC_8classes 
#
#    #_WithCasing_WOTurnToken_CombineUCandMC_8classes
#    #_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_${dataset_seed}
#
#    #_WOTurnToken_CombineUCandMC_8classes
#    #_WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
#
#    # _WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
#    # _WithCasing_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
#
#    # _WithCasing_WOTurnToken_CombineUCandMC_8classes
#    # _WithPunct_WOTurnToken_CombineUCandMC_8classes
#
#    #_WOTurnToken_CombineUCandMC_8classes
#    #_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
#    #_WOTurnToken_CombineUCandMC_4classes
#    #_WOTurnToken_CombineUCandMC_4classes_9168TrainSamples
#
#    do
#        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix}
#        bash run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset
#    done
#    echo $loss_wts
#done
#done
#
#exit


train_mode=TE
for HF_model_name in bert-base-uncased #bert-base-cased bert-base-uncased
do
for loss_wts in  1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9  #  1_0.1 1_0.25 1_0.5 1_0.75 1_1 0.1_1 0.25_1 0.5_1 0.75_1 

# 0.75_0.25 # 1_0 0_1 0.5_0.5 0.25_0.75  0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
# 0.95_0.05
do
    seed=42
    grid=True #False #True
    gpu=1
    dataset=fisher #guetenburg
    #dataset=fisher
    challenge_eval=0
    eval_dataset=fisher
    for dataset_seed in 42 43 44 45 46
    do
    for num_samples in 10 #50 100 250 500 # 5000 9168
    do
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_${dataset_seed}

    #_WOTurnToken_CombineUCandMC_8classes
    #_WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples

    # _WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
    # _WithCasing_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples

    # _WithCasing_WOTurnToken_CombineUCandMC_8classes
    #_WithPunct_WOTurnToken_CombineUCandMC_8classes
    # _WithCasing_WOTurnToken_CombineUCandMC_8classes
    #_WOTurnToken_CombineUCandMC_8classes
    #_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples
    #_WOTurnToken_CombineUCandMC_4classes
    #_WOTurnToken_CombineUCandMC_4classes_9168TrainSamples

    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT${common_suffix}
        bash run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset
    done
    done
    done
    echo $loss_wts
done
done

exit

train_mode=TE #TE

for seed in 42 #43 44 # 42
do
for loss_wts in 1_0 0_1 0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
# 0.95_0.05
do
    grid=True #False #True
    gpu=1
    challenge_eval=0
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes # _CombineUCandMC_8classes
    #_CombineUCandMC _WOTurnToken_CombineUCandMC
    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif${common_suffix}
        bash run_TrueCasing_data_v3.sh 1 $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts
    done

    echo $loss_wts
done
done



exit


