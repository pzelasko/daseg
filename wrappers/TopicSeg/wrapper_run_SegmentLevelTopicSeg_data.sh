
set -x

############################################################

#######################   sequence level binary classification ######################

train_mode=TE #E # TE
for HF_model_name in dkleczek/bert-base-polish-cased-v1 # bert-base-multilingual-cased # bert-base-uncased #bert-base-cased bert-base-uncased
do
for loss_wts in 1_0 #1_0 
do
for dataset in logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_ logisfera_telco_CombineAsItIs_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_

# logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_5times_WithOrigData_
#logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_ telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_ 

#energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_ logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_ telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_


#logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_

# logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_

#logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_
#energa_logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ # energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_

#energa_ForI-Scheme_WithSEP_OnlySegmentation_ energa_ForI-Scheme_WithSEP_  energa_ForI-Scheme_ #energa_ForI-Scheme_WithSEP_
#energa_logisfera_
#energa_logisfera_ energa_telco_ logisfera_telco_  energa_logisfera_telco_ energa_ logisfera_ telco_ 
do
    seed=42
    grid=False #True #False #True
    gpu=1
    gpu_ind=7
    #dataset=fisher
    challenge_eval=0
    eval_dataset=energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_
    for cv in $(seq 1 10)
    do
        for common_suffix in None
        do
            task=topicseg_Morethan2TasksArch_SeqClassif_BERT
            bash wrappers/TopicSeg/run_TopicSeg_data.sh $cv $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset $gpu_ind
        done
    done
    echo $loss_wts
done
done
done


