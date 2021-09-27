
set -x

############################################################

##################   token level classificayion ####################################

train_mode=E # TE
for HF_model_name in dkleczek/bert-base-polish-cased-v1 # bert-base-multilingual-cased # bert-base-uncased #bert-base-cased bert-base-uncased
do
for loss_wts in 1_0 
do
for dataset in logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_
#energa_logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ # energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_

#energa_ForI-Scheme_WithSEP_OnlySegmentation_ energa_ForI-Scheme_WithSEP_  energa_ForI-Scheme_ #energa_ForI-Scheme_WithSEP_
# data_${i}_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/
#energa_logisfera_
#energa_logisfera_ energa_telco_ logisfera_telco_  energa_logisfera_telco_ energa_ logisfera_ telco_ 
do
    seed=42
    grid=False #True #False #True
    gpu=1
    gpu_ind=6
    #dataset=fisher
    challenge_eval=0 #1
    eval_dataset=energa_ForI-Scheme_WithSEP_OnlySmoothSegmentation_
    for cv in 1 #$(seq 2 10)
    do
        for common_suffix in None
        do
            task=truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT
            bash wrappers/TopicSeg/run_TopicSeg_data.sh $cv $train_mode $gpu 4 8 -1 $seed $task $grid $challenge_eval $common_suffix $loss_wts $dataset $HF_model_name $eval_dataset $gpu_ind
        done
    done
    echo $loss_wts
done
done
done


