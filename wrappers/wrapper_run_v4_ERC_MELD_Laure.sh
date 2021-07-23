

set -x

source /home/rpapagari/.bashrc

source activate daseg_v2

train_mode=TE

for concat_aug in 0 -1 ##-1 #-1 #-1 # 0 #-1
do
for emotion_sentiment in Emotion Sentiment
do
for task_name in MELD_${emotion_sentiment}_ConvClassif_Longformer_fs100ms MELD_${emotion_sentiment}_ConvClassif_bilstm_fs100ms MELD_${emotion_sentiment}_ConvClassif_ResNet_fs10ms 


#MELD_${emotion_sentiment}_ConvClassif_xformer_fs10ms_spkrind MELD_${emotion_sentiment}_ConvClassif_xformer_fs10ms MELD_${emotion_sentiment}_ConvClassif_XFormerAddSpeakerEmb_fs10ms MELD_${emotion_sentiment}_ConvClassif_XFormerConcatSpeakerEmb_fs10ms
#MELD_${emotion_sentiment}_UttClassif_bilstm_fs100ms MELD_${emotion_sentiment}_UttClassif_ResNet_fs10ms MELD_${emotion_sentiment}_UttClassif_longformer_fs100ms
do
    gpu=1
    grid=True #False #True
    noise_aug=True
    main_exp_dir=/export/c01/lmorove1/daseg_erc/daseg/ERC_MELD_expts
    bash run_v4_ERC_MELD.sh 1 $train_mode $gpu 6 5 $concat_aug 42 $grid $task_name $emotion_sentiment $noise_aug $main_exp_dir
    

done
done
done




