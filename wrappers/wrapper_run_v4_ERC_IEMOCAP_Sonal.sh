

set -x
source /home/rpapagari/.bashrc
source activate daseg_v2



train_mode=TE
for concat_aug in 0 #0 -1 ##-1 #-1 #-1 # 0 #-1
do
for task_name in IEMOCAP_ConvClassif_XFormerConcatAveParamSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerConcatAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerAddAveSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
# IEMOCAP_ConvClassif_xformer_smoothed_overlap_silenceNone_OOSNone
#IEMOCAP_ConvClassif_XFormerConcatSpeakerEmb_smoothed_overlap_silenceNone_OOSNone 
#IEMOCAP_ConvClassif_XFormerAddSpeakerEmb_smoothed_overlap_silenceNone_OOSNone
do
    for cv in 1 2 3 4 5 
    do
        gpu=1
        grid=True
        bash run_v4_ERC_IEMOCAP_Sonal.sh $cv $train_mode $gpu 6 5 $concat_aug 42 $grid $task_name
    done
done
done



