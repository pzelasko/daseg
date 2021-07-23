
set -x

#train_mode=E
#seed=42
#for common_suffix in None
#do
#    if [[ $common_suffix != None ]]
#    then
#        task_name=punctuation_LastSubTokenLossCompute_longformer_tokenclassif${common_suffix}
#    else
#        task_name=punctuation_LastSubTokenLossCompute_longformer_tokenclassif
#    fi
#    loss_wts=1_1
#    grid=True
#    gpu=1
#    challenge_eval=0
#    batch_size=8
#    gacc=4
#    bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu $batch_size $gacc -1 $seed $task_name $grid $challenge_eval $common_suffix $loss_wts
#
#done


###########################
#
#train_mode=TE
#seed=42
#for common_suffix in _WOCasingWOTurnToken_8classes _WOCasing_8classes _WOTurnToken_8classes _8classes
## None _WOCasing _WOTurnToken _WOCasingWOTurnToken
#do
#    if [[ $common_suffix != None ]]
#    then
#        task_name=punctuation_longformer_tokenclassif${common_suffix}
#    else
#        task_name=punctuation_longformer_tokenclassif
#    fi
#    loss_wts=1_1
#    grid=True
#    gpu=1
#    challenge_eval=0
#    batch_size=8 # 6
#    gacc=4 # 5
#    bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu $batch_size $gacc -1 $seed $task_name $grid $challenge_eval $common_suffix $loss_wts
#
#done
#
#exit

############################################################
#
#train_mode=TE
#
#for seed in 42 #43 44 # 42
#do
#for loss_wts in 1_0 0_1 #0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
#do
#    grid=True
#    gpu=1
#    challenge_eval=0
#    for common_suffix in _WOTurnToken_CombineUCandMC_8classes 
#    #_CombineUCandMC _WOTurnToken_CombineUCandMC
#    do
#        for task in truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_1Att1ClassifLinearLayers${common_suffix} #truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers${common_suffix}
#        do
#            batch_size=4
#            gacc=8
#            bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu $batch_size $gacc -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts
#        done
#    done
#
#    echo $loss_wts
#done
#
#done
#
#exit




train_mode=E

for seed in 42 #43 44 # 42
do
for loss_wts in 0.5_0.5 #0.9_0.1 # 1_0 0_1 #0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9   ## 0.5_1 1_0.5 1_0.1 0.1_1
do
    grid=True #False #True
    gpu=1
    challenge_eval=0
    for common_suffix in _WOTurnToken_CombineUCandMC_8classes #_CombineUCandMC_8classes
    #_CombineUCandMC _WOTurnToken_CombineUCandMC
    do
        task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif${common_suffix}
        bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu 8 4 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts
    done

    echo $loss_wts
done

done

exit

##########################################################


seed=42
train_mode=TE
grid=True
gpu=1
challenge_eval=0
for task in truecasingWOpunct_longformer_tokenclassif_CombineUCandMC
#truecasingWOTurnToken_longformer_tokenclassif_CombineUCandMC truecasing_longformer_tokenclassif_CombineUCandMC truecasingWOpunct_longformer_tokenclassif_CombineUCandMC truecasingWOpunctWOTurnToken_longformer_tokenclassif_CombineUCandMC
do
    bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu 8 4 -1 $seed $task $grid $challenge_eval _CombineUCandMC 1_1
done

exit

for task in  truecasingWOTurnToken_longformer_tokenclassif truecasing_longformer_tokenclassif truecasingWOpunct_longformer_tokenclassif truecasingWOpunctWOTurnToken_longformer_tokenclassif
do
    bash run_TrueCasing_data_v2.sh 1 $train_mode $gpu 8 4 -1 $seed $task $grid $challenge_eval None 1_1
done


exit

#########################################################

