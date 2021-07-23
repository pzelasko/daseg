



for results_type in results
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_-1_warmup200steps/longformer_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_-1_warmup200steps/longformer_IEMOCAP_v2_42/
    eval_script=metrics_for_comfort_v2.py
    collar=0

    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_-1/longformer_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_-1/longformer_IEMOCAP_v2_42/
    eval_script=metrics_for_comfort_v2.py
    collar=0

    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_-1_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_-1_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    eval_script=metrics_for_comfort_v2.py
    collar=0

    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs3_gacc10_concat_aug_-1_warmup200steps/longformer_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs3_gacc10_concat_aug_-1_warmup200steps/longformer_IEMOCAP_v2_42/
    eval_script=metrics_for_comfort_v2.py
    collar=0

    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs24_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs24_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    collar=0

    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_bilstm/bilstm_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_bilstm/bilstm_IEMOCAP_v2_42/
    collar=0

    results_type=results_conversations
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_bilstm/bilstm_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_bilstm/bilstm_IEMOCAP_v2_42/
    collar=0

    results_type=results_conversations
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs24_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs24_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/
    collar=0

    python $eval_script $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
done


exit





for concat_aug in -1 
do
for max_len in 4000 
do
for results_type in results 
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_MMSE/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_MMSE/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py 
    
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_AlignmentsSpkrOnlyINVPAR_v2_NoSEP/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_AlignmentsSpkrOnlyINVPAR_v2_NoSEP/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py 
    
    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_AlignmentsSpkrOnlyINVPAR_v2_NoSEP_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_AlignmentsSpkrOnlyINVPAR_v2_NoSEP_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v6_MMSE/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v6_MMSE/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7_MMSE/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7_MMSE/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v6/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v6/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py

    task_suffix=_MMSE #''
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7_Cased${task_suffix}/bert-base-cased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v7_Cased${task_suffix}/bert-base-cased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo${task_suffix}.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_2_1_Wts_transcripts_v7/TransformerMultiModalMultiLossSeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_2_1_Wts_transcripts_v7/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py

    for collar in 0 
    do
        python $eval_script $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done

exit




for concat_aug in -1 
do
for max_len in 4000 
do
for results_type in results 
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_MMSE/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_MMSE/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py 
    
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_MMSE/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_MMSE/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_NoSEP/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_NoSEP/bert-base-uncased_ADReSSo_42/
    eval_script=metrics_for_comfort_v2_ADReSSo.py
    
    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_NoSEP_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs_NoSEP_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py




    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py



    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v2/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v2/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v2_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v2_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v4/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v4/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v4_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v4_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v5/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v5/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v5_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v5_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRange_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRange_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRangeV2_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRangeV2_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    #ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRangeV3_MMSE/bert-base-uncased_ADReSSo_4,/
    #op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_transcripts_v3_SimilarDynamicRangeV3_MMSE/bert-base-uncased_ADReSSo_42/
    #eval_script=metrics_for_comfort_v2_ADReSSo_MMSE.py

    

    for collar in 0 
    do
        python $eval_script $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done

exit




for concat_aug in -1 
do
for max_len in 4000 
do
for results_type in results 
do
for noise_aug in NoiseAug_ ''
do
for resnet_type in SoTAEnglishXVectorCleanTrue SoTAEnglishXVectorTrue
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_PARaudio_MMSE/resnet_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_PARaudio_MMSE/resnet_SeqClassification_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_MMSE_L1Loss/resnet_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_MMSE_L1Loss/resnet_SeqClassification_ADReSSo_42/

    
    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo_MMSE.py $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_AlternatingTokenTypeEmbed/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_AlternatingTokenTypeEmbed/bert-base-uncased_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_NoTokenTypeEmbed/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_NoTokenTypeEmbed/bert-base-uncased_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_1TokenTypeEmbedForINV/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSilSpkr_v1_1TokenTypeEmbedForINV/bert-base-uncased_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_50_vggish_batchnormiplayer/bilstm_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_50_vggish_batchnormiplayer/bilstm_SeqClassification_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_50_vggish_batchnormiplayer/bilstm_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_50_vggish_batchnormiplayer/bilstm_SeqClassification_ADReSSo_42/

    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_200_vggish_batchnormiplayer_meanpooling/bilstm_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_200_vggish_batchnormiplayer_meanpooling/bilstm_SeqClassification_ADReSSo_42/

     ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_200_vggish_batchnormiplayer_meanpooling_2BiLSTMlayers/bilstm_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_BiLSTM_frame_len1_UttClassif_bs64_gacc1_concat_aug_-1_model_TrainDevTest_maxlen_300_Epochs_200_vggish_batchnormiplayer_meanpooling_2BiLSTMlayers/bilstm_SeqClassification_ADReSSo_42/
    
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR/bert-base-uncased_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR/bert-base-uncased_ADReSSo_42/


    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done

exit





for concat_aug in -1 0
do
for max_len in 4000 
do
for results_type in results 
do
for noise_aug in NoiseAug_ ''
do
for resnet_type in SoTAEnglishXVectorCleanTrue SoTAEnglishXVectorTrue
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_PARaudio/resnet_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_${resnet_type}_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_${noise_aug}maxlen_8000_Epochs_50_PARaudio/resnet_SeqClassification_ADReSSo_42/

    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done
done
done
exit








for concat_aug in -1 0
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_4,/
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/


    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir/${results_type}.pkl $collar > ${op_file}/diarize_${results_type}.txt_collar${collar}
    done
done
done
done

exit



for concat_aug in -1 0
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1 0
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorCleanTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit




for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_10_1_Wts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_10_1_Wts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit




for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_Alignments_v1/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done


exit




for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_2_1_Wts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_2_1_Wts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit


for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_EqualWts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModalMultiLoss_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_EqualWts/TransformerMultiModalMultiLossSeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2_ADReSSo.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit


for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsNoSil_v1/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done


for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_0.1Warmup/longformer_text_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_0.1Warmup/longformer_text_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_BERT_0.1Warmup/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_BERT_0.1Warmup/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_0.1Warmup/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_0.1Warmup/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainTest/longformer_text_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainTest/longformer_text_SeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_BERT/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainTest_BERT/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_speechatt/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers_speechatt/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.1_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_2CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit




for concat_aug in -1 0
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_8000_Epochs_50/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in -1
do
for max_len in 4000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT/bert-base-uncased_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT/bert-base-uncased_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



#for concat_aug in -1
#do
#for max_len in 4000 
#do
#for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
#do
#    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_3CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
#    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_1000_Epochs_100_3CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt
#
#
#    for collar in 0 
#    do
#        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
#    done
#done
#done
#done
#
#exit


#for concat_aug in -1
#do
#for max_len in 4000 
#do
#for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
#do
#    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_2000_Epochs_100_1CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
#    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_2000_Epochs_100_1CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt
#
#
#    for collar in 0 
#    do
#        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
#    done
#done
#done
#done
#
#exit

for concat_aug in -1
do
for max_len in 1000  # 2000
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_${max_len}_Epochs_100_1CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_frame_len0.01_MultiModal_bs6_gacc5_concat_aug_-1_maxlenspeech_${max_len}_Epochs_100_1CrossAttLayers/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit




no_epochs=100
for concat_aug in -1
do
for max_len in 4000 5000
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_MultiModal_maxlenspeech_${max_len}_Epochs_100/TransformerMultiModalSeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_MultiModal_maxlenspeech_${max_len}_Epochs_100/TransformerMultiModalSeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in 0 -1
do
for max_len in 8000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}_spkrPAR/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}_spkrPAR/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in 0 -1
do
for max_len in 8000 
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}_spkrPAR_Epochs_50/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}_spkrPAR_Epochs_50/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in 0 -1
do
for max_len in 8000 4000
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_${max_len}/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_NoiseAug_maxlen_${max_len}/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done

exit



for concat_aug in 0 -1
do
for max_len in 8000 4000
do
for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_${concat_aug}_model_TrainDevTest_maxlen_${max_len}/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done
done
done
exit


for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar #> ${op_file}_collar${collar}
    done
done

exit

for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest_NoiseAug/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest_NoiseAug/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar # > ${op_file}_collar${collar}
    done
done

exit



for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest_NoiseAug/resnet_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest_NoiseAug/resnet_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar # > ${op_file}_collar${collar}
    done
done

exit




for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest_NoiseAug/longformer_speech_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_0_model_TrainDevTest_NoiseAug/longformer_speech_SeqClassification_ADReSSo_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar # > ${op_file}_collar${collar}
    done
done

exit





for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest/longformer_speech_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_model_TrainDevTest/longformer_speech_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest/longformer_text_SeqClassification_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest/longformer_text_SeqClassification_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=ADReSSo_IS2021_expts/ADReSSo_CV_,_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest/longformer_text_ADReSSo_42/${results_type}.pkl
    op_file=ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest/longformer_text_ADReSSo_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit

for results_type in results_with_uttids #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar #> ${op_file}_collar${collar}
    done
done

for results_type in results_with_uttids #results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar #> ${op_file}_collar${collar}
    done
done


exit





for results_type in results_with_uttids
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformer_cnnop_segpool_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind_PoolSegments/xformer_cnnop_segpool_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformer_cnnop_segpool_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind_PoolSegments/xformer_cnnop_segpool_IEMOCAP_v2_42/diarize_${results_type}.txt


    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit





for results_type in results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs3_gacc10_concat_aug_-1/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs3_gacc10_concat_aug_-1/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit





for results_type in results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

for results_type in results_with_uttids_SpkrF results_with_uttids_SpkrM #results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


exit

for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit





for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformer_cnnop_segpool_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments/xformer_cnnop_segpool_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformer_cnnop_segpool_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments/xformer_cnnop_segpool_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone/xformer_IEMOCAP_v2_42/${results_type}.pkl

    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silenceNone_OOSNone_NeutralNone/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit





for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit





for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformersegpool_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments/xformersegpool_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_NoiseAug_frame_len0.01_ConvClassif_bs6_gacc5_CP_0_xformersegpool_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind_PoolSegments/xformersegpool_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


####################

for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs15_gacc2_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs15_gacc2_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOSNone_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit

###########################


for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs15_gacc2_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs15_gacc2_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit



for results_type in results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


for results_type in results_with_uttids_InferenceWOspkrind #results_with_uttids #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS_spkrind/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS_spkrind/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


for results_type in results #results_conversations # results
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_warmup200steps/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs24_gacc5_concat_aug_0_warmup200steps/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


exit


for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/${results_type}_with_uttids.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt_temp

    for collar in 0 #6 
    do
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
        python metrics_for_comfort_v3.py $ip_dir $collar > ${op_file}_collar${collar}_label_NOinterest_overlap
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}_IgnoreBothSilOOS
        
    done
done

exit




for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_silence_OOS/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 #6 
    do
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}_IgnoreOnlySil
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}_IgnoreBothSilOOS
        
    done
done

exit




for results_type in results_conversations # results
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_UttClassif_bs6_gacc5_concat_aug_-1_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit


for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_NoSilence_OOS/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_NoSilence_OOS/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}_IgnoreOnlySil
    done
done

exit



for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_NoSilence_AllClasses/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048_smoothed_overlap_NoSilence_AllClasses/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 6 
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
        #python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}_IgnoreOnlySil
    done
done

exit





for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 6 #0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit




for results_type in results #results_conversations
do
    frame_len=0.08
    non_zero_collar=6 # 0.5/frame_len
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentationSoTAEnglishXVector_IsolatedUttTraining_run1_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.08_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_longformer/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentationSoTAEnglishXVector_IsolatedUttTraining_run1_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.08_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_longformer/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in $non_zero_collar # 0
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

exit






for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt

    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done






for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done




for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorFalse_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorFalse_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs6_gacc5_concat_aug_0_ResNet_maxseqlen_2048/ResNet_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done




for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorFalse_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorFalse_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done





for results_type in results #results_conversations
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_-1_bilstm/bilstm_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_UttClassif_bs6_gacc5_concat_aug_-1_bilstm/bilstm_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done



for results_type in results #results_ASHNF_UttEval_fs100ms
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs6_gacc5_concat_aug_-1_bilstm/bilstm_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs6_gacc5_concat_aug_-1_bilstm/bilstm_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


for results_type in results #results_ASHNF_UttEval_fs100ms #results
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs10_gacc5_concat_aug_1_warmup200steps/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs10_gacc5_concat_aug_1_warmup200steps/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 #5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


for results_type in results #results_ASHNF_UttEval_fs100ms #results results_ASHNF_UttEval_fs100ms
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS_labelsmoothing0.1/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS_labelsmoothing0.1/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 #5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done

for results_type in results #results_ASHNF_UttEval_fs100ms #results results_ASHNF_UttEval_fs100ms
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS_labelsmoothing0.05/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS_labelsmoothing0.05/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 #5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done


for results_type in results #results_ASHNF_UttEval_fs100ms #results results_ASHNF_UttEval_fs100ms
do
    ip_dir=journal_v2/IEMOCAP_v2_CV_,_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS/longformer_IEMOCAP_v2_42/${results_type}.pkl
    op_file=journal_v2/IEMOCAP_v2_CV_1_ExactLabelScheme_smoothSegmentation_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.1_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_smoothed_overlap_silence_OOS/longformer_IEMOCAP_v2_42/diarize_${results_type}.txt
    for collar in 0 #5
    do
        python metrics_for_comfort_v2.py $ip_dir $collar > ${op_file}_collar${collar}
    done
done



