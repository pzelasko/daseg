
train_mode=TE
for concat_aug in -1 #0 # -1 ##-1 #-1 #-1 # 0 #-1
do
for task_name in IEMOCAP_ConvClassif_BERT

#IEMOCAP_UttClassif_bilstm 
#IEMOCAP_UttClassif_ResNet
#IEMOCAP_UttClassif_longformer

# IEMOCAP_ConvClassif_XFormerConcatTokenTypeEmb_smoothed_overlap_silenceNone_OOSNone_spkrind
#IEMOCAP_ConvClassif_XFormerAddAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone_spkrind
#IEMOCAP_ConvClassif_XFormerConcatAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone_spkrind
#IEMOCAP_ConvClassif_xformer_NoInfreqEmo
#IEMOCAP_ConvClassif_XFormerConcatAveParamSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerConcatAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerAddAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
# IEMOCAP_ConvClassif_xformer_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerConcatSpeakerEmb_smoothed_overlap_silenceNone_OOSNone 
#IEMOCAP_ConvClassif_XFormerAddSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
do
    for cv in 2 3  4 #1 2 3 4 5 
    do
        gpu=1
        grid=True
        bash run_v4_ERC_IEMOCAP.sh $cv $train_mode $gpu 3 10 $concat_aug 42 $grid $task_name
        #bash run_v4_ERC_IEMOCAP.sh $cv $train_mode $gpu 6 5 $concat_aug 42 $grid $task_name
        #bash run_v4_ERC_IEMOCAP.sh $cv $train_mode $gpu 100 5 $concat_aug 42 $grid $task_name
    done
done
done



