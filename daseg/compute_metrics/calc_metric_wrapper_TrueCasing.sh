

tmpfile=$(mktemp tmp/eval_script.XXXXXX)

for label_block_name in 1 0 # 1
do
for num_samples in 10 #50 100 250 500 1000 5000 9168  # $1 # 250 #5000 #1000 #500 1000 #9168
do
for loss_wts in   1_0 0.9_0.1 0.75_0.25 0.5_0.5 0.25_0.75 0.1_0.9  0_1 #   1_0 0_1 0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 # 0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in resultsfisher #results # resultsguetenburg # resultsfisher # resultsearning21_benchmark results
do
    #ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_4,_loss_wts_${loss_wts}_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    #op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_42_loss_wts_${loss_wts}_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    
    #ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_fromCNNcased_crossdomain_truecasing_punctuation_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_4,_loss_wts_${loss_wts}_bert-base-uncased_CNNBERTcased2Fisher/crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    #op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_fromCNNcased_crossdomain_truecasing_punctuation_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_42_loss_wts_${loss_wts}_bert-base-uncased_CNNBERTcased2Fisher/crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/


    #ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_fromCNN2500cased_crossdomain_truecasing_punctuation_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_4,_loss_wts_${loss_wts}_bert-base-uncased_CNNBERTcased2Fisher/crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    #op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_fromCNN2500cased_crossdomain_truecasing_punctuation_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_${num_samples}_subsetseed_42_loss_wts_${loss_wts}_bert-base-uncased_CNNBERTcased2Fisher/crossdomain_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/


    ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_2500_subsetseed_42_loss_wts_${loss_wts}_Epochs_50_bert-base-cased_CNN/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WOTurnToken_CombineUCandMC_8classes_2500_subsetseed_42_loss_wts_${loss_wts}_Epochs_50_bert-base-cased_CNN/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/



    #echo $label_block_name $num_samples $loss_wts `grep macro_f1 ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name}`
    #echo $label_block_name $num_samples $loss_wts `grep classwise_f1 ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name}`

    #echo $ip_dir 

    #ip_dir=$PWD/$ip_dir
    #op_file=$PWD/$op_file
    eval_script=$PWD/metrics_for_comfort_TrueCasingPunctuation_BigData.py

    echo "source activate daseg_v2 && python $eval_script $ip_dir/${results_type}.pkl $label_block_name ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name}" >> $tmpfile
done
done
done
done

#exit

echo tmpfile is $tmpfile
log_dir="$(basename -- $tmpfile)"
echo log_dir is $log_dir
bash /home/rpapagari/submission_script.sh $tmpfile tmp/${log_dir}.log/ 100 6G


exit



tmpfile=$(mktemp tmp/eval_script.XXXXXX)

for loss_wts in 0.1_0.9 # 1_0 0_1 0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 # 0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in results #resultsfisher #results # resultsguetenburg # resultsfisher # resultsearning21_benchmark results
do
for label_block_name in 0 1
do

    ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithCasing_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples_loss_wts_0_1_Epochs_25_bert-base-uncased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithCasing_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples_loss_wts_0_1_Epochs_25_bert-base-uncased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/

    ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithCasing_WOTurnToken_CombineUCandMC_8classes_loss_wts_0_1_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithCasing_WOTurnToken_CombineUCandMC_8classes_loss_wts_0_1_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    

    ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithPunct_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/
    op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithPunct_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25_bert-base-uncased/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42/


    ip_dir=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples_loss_wts_1_0_Epochs_25_bert-base-uncased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42
    op_file=/export/fs02/rpapagari/TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_WithPunct_WOTurnToken_CombineUCandMC_8classes_9168TrainSamples_loss_wts_1_0_Epochs_25_bert-base-uncased_guetenburg/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_BERT_TrueCasing_42

    echo $ip_dir 

    #ip_dir=$PWD/$ip_dir
    #op_file=$PWD/$op_file
    eval_script=$PWD/metrics_for_comfort_TrueCasingPunctuation_BigData.py

    echo "source activate daseg_v2 && python $eval_script $ip_dir/${results_type}.pkl $label_block_name ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name}" >> $tmpfile
done
done
done

echo tmpfile is $tmpfile
log_dir="$(basename -- $tmpfile)"
echo log_dir is $log_dir
bash /home/rpapagari/submission_script.sh $tmpfile tmp/${log_dir}.log/ 100 6G &

exit




for loss_wts in 1_0 0_1 0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 # 0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in resultsearning21_benchmark  #  results # resultsearning21_benchmark results
do
for label_block_name in 0 1
do
    for model_name  in truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes
    # truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_CombineUCandMC_8classes
    do

    ip_dir=TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}_Epochs_25/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/
    op_file=TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}_Epochs_25/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/

    eval_script=metrics_for_comfort_TrueCasingPunctuation.py


    python $eval_script $ip_dir/${results_type}.pkl $label_block_name > ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name} &
done
done
done
done

exit



for loss_wts in 1_0 0_1 # 0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 #0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in results # resultsearning21_benchmark results
do
for label_block_name in 0 1
do
    for model_name  in truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes # truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_CombineUCandMC_8classes
    do
    
    ip_dir=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}_Epochs_15/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/
    op_file=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}_Epochs_15/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/

    eval_script=metrics_for_comfort_TrueCasingPunctuation.py


    python $eval_script $ip_dir/${results_type}.pkl $label_block_name > ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name} &
done
done
done
done

exit



for loss_wts in  0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 #0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in results # resultsearning21_benchmark results
do
for label_block_name in 0 1
do
    for model_name  in truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_CombineUCandMC_8classes
    do
    
    ip_dir=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/
    op_file=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/

    eval_script=metrics_for_comfort_TrueCasingPunctuation.py

    python $eval_script $ip_dir/${results_type}.pkl $label_block_name > ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name} &
done
done
done
done

exit


for loss_wts in  0.5_0.5 0.25_0.75 0.75_0.25 0.9_0.1 0.1_0.9 #0.1_1 1_0.1 1_1 1_0 0_1 1_0.5 0.5_1
do
for results_type in results # resultsearning21_benchmark results
do
for label_block_name in 0 1
do
    for model_name  in truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers_WOTurnToken_CombineUCandMC_8classes
    do
    
    ip_dir=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers_TrueCasing_42/
    op_file=TrueCasing_expts_data_v2/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_${model_name}_loss_wts_${loss_wts}/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_3ClassifLinearLayers_TrueCasing_42/


    eval_script=metrics_for_comfort_TrueCasingPunctuation.py
    
    python $eval_script $ip_dir/${results_type}.pkl $label_block_name > ${op_file}/summary_${results_type}.txt_label_block_name${label_block_name} &
done
done
done
done

exit




for results_type in resultsearning21_benchmark #results
do
    
    ip_dir=TrueCasing_expts/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasingWOpunct_longformer_tokenclassif_NoLowerTurnToken/truecasing_longformer_tokenclassif_TrueCasing_42/

    #ip_dir=TrueCasing_expts/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasingWOpunctWOTurnToken_longformer_tokenclassif_NoLowerTurnToken/truecasing_longformer_tokenclassif_TrueCasing_42/

    #ip_dir=TrueCasing_expts/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasingWOTurnToken_longformer_tokenclassif_NoLowerTurnToken/truecasing_longformer_tokenclassif_TrueCasing_42/ 

    #ip_dir=TrueCasing_expts/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_longformer_tokenclassif_NoLowerTurnToken/truecasing_longformer_tokenclassif_TrueCasing_42/

    op_file=$ip_dir/
    eval_script=metrics_for_comfort_v2.py
    collar=0

    python $eval_script $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
done

exit


