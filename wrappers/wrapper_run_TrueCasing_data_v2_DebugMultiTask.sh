
set -x



#############################################################

train_mode=TE

for seed in 42 #43 44 # 42
do
for loss_wts in 1_0    
do
    grid=True
    gpu=1
    challenge_eval=0
    for task in truecasing_punctuation_Morethan2Tasks_longformer_tokenclassif_CombineUCandMC
    do
        bash run_TrueCasing_data_v2_DebugMultiTask.sh 1 $train_mode $gpu 8 4 -1 $seed $task $grid $challenge_eval _CombineUCandMC $loss_wts
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
for task in Custom_truecasingWOpunct_longformer_tokenclassif_CombineUCandMC
#MultitaskModelDebug_truecasingWOpunct_longformer_tokenclassif_CombineUCandMC 
#truecasingWOpunct_longformer_tokenclassif_CombineUCandMC
do
    bash run_TrueCasing_data_v2_DebugMultiTask.sh 1 $train_mode $gpu 8 4 -1 $seed $task $grid $challenge_eval _CombineUCandMC 1_1
done

exit
#########################################################


