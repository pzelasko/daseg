

#for i in energa #logisfera telco 
#do
#    python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/ 10 $i I- True True True
#done


#python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_energa_logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/ 10 energa,logisfera,telco I- True True True

#python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_logisfera_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/ 10 logisfera,telco I- True True True

#python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_logisfera_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/ 10 logisfera I- True True True
#python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_telco_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10/ 10 telco I- True True True

#for  i in energa logisfera telco
#do
#    python topic_seg_utils/prepare_data_v3.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_WithSEP_cv10/ 10 $i Exact True False False
#done

####  create concataug data
for fold in $(seq 1 10)
do
    for  i in energa logisfera telco
    do
        concat_aug_dir=/dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_cv10/

        #python topic_seg_utils/prepare_concataug_data.py ${concat_aug_dir} $fold /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_WithSEP_cv10/cv_${fold}/train.tsv I- False True True
        #cp ${concat_aug_dir}/cv_${fold}/utt2csvpath_train ${concat_aug_dir}/cv_${fold}/utt2csvpath
        #cat /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_WithSEP_cv10//cv_${fold}/utt2csvpath_* > /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_WithSEP_cv10//cv_${fold}/utt2csvpath

        python topic_seg_utils/concat_data_dir.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_OnlySmoothSegmentation_cv10//cv_${fold}/,${concat_aug_dir}//cv_${fold}/ /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_OnlySmoothSegmentation_ConcatAug_2times_WithOrigData_cv10//cv_${fold}/ 0
    
    done
done


exit


for i in energa logisfera telco 
do
    python topic_seg_utils/prepare_data_v2.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_cv10/ 10 $i I- False False ; 
    python topic_seg_utils/prepare_data_v2.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_cv10/ 10 $i I- True False; 
    python topic_seg_utils/prepare_data_v2.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_ForI-Scheme_WithSEP_OnlySegmentation_cv10/ 10 $i I- True True; 
    
done


#for i in energa logisfera telco; do 
#    python topic_seg_utils/prepare_data.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}_cv10/ 10 $i; 
#done
#
#
#python topic_seg_utils/prepare_data.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_energa_logisfera_cv10/ 10 energa,logisfera
#
#
#python topic_seg_utils/prepare_data.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_energa_telco_cv10/ 10 energa,telco
#
#
#python topic_seg_utils/prepare_data.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_logisfera_telco_cv10/ 10 logisfera,telco
#
#
#python topic_seg_utils/prepare_data.py /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_energa_logisfera_telco_cv10/ 10 energa,logisfera,telco


#for i in energa_logisfera_ energa_telco_ logisfera_telco_  energa_logisfera_telco_ energa_ logisfera_ telco_
#do
#    #cat /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}cv10/cv_1/*.tsv > /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}cv10/all.tsv
#    cp /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}cv10/cv_1/utt2csvpath  /dih4/dih4_2/jhu/Raghu/topic_seg_data_dirs/data_${i}cv10/utt2csvpath
#done



