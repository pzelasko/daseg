from more_itertools import flatten
import sklearn.metrics as sklmetrics
from daseg.metrics_v2 import compute_sklearn_metrics, compute_zhao_kawahara_metrics_speech, conv_level_emo_pred, conv_level_senti_pred
import pickle as pkl
import os, sys
import glob
import numpy as np
import pandas as pd


def calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files):
    conf_mat = sklmetrics.confusion_matrix(y_true_all, y_pred_all, labels=labels_list)
    conf_mat = conf_mat/no_results_files
    conf_mat = conf_mat.astype('int')
    classwise_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average=None)
    micro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='weighted')
    macro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='macro')
    unweighted_acc = sklmetrics.accuracy_score(y_true_all, y_pred_all)
    classif_report = sklmetrics.classification_report(y_true_all, y_pred_all, labels=labels_list)
    
    #print(metrics_list)
    metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2]
    metrics_list = metrics_list.split(',')
    for metric in metrics_list:
        
        if metric == 'confusion':
            print(conf_mat)
            print(labels_list)
        if metric == 'f1-score':
            print(f'classwise_f1: {classwise_f1}')
            print(f'micro_f1: {micro_f1}')
            print(f'macro_f1: {macro_f1}')
        if metric == 'accuracy':
            print(f'unweighted_acc: {unweighted_acc}')                    
    print(classif_report)


def filter_results(y_true, y_pred, utt_id_list, data_dir, label_interest=['overlap']):
    merge_emotions =  {'exc':'hap', 'xxx':'OOS', 'fea':'OOS',
                        'oth':'OOS', 'sur':'OOS', 'dis':'OOS',
                        'sil':'silence'}
    new_y_true = []
    new_y_pred = []
    utt_id_list = np.concatenate(utt_id_list)
    test_tsv = pd.read_csv(data_dir + '/test.tsv', sep=',', header=None)
    uttid2labelspath = dict(zip(test_tsv[0], test_tsv[1]))
    frame_len = 0.01
    subsampling_factor = 8
    max_seq_len = 2048
    for ind,utt_id in enumerate(utt_id_list):
        utt_labels_fromdataloader = y_true[ind]
        utt_model_labels = y_pred[ind]
        seg_labels_path = uttid2labelspath[utt_id]        
        new_labels = []
        if os.path.isfile(seg_labels_path):
            ### for ERC task
            seg_labels = pd.read_csv(seg_labels_path, sep=',')
            seg_labels = seg_labels.values.tolist()
            for i in seg_labels:
                if len(i) == 4:
                    begin_time, end_time, emo_label, spkr_ind = i[0], i[1], i[2], i[3]
                else:
                    begin_time, end_time, emo_label = i[0], i[1], i[2]
                    spkr_ind = '0'
                emo_label_original = emo_label
                seg_count_frames = round((end_time - begin_time) / frame_len)
                
                if emo_label in merge_emotions:
                    emo_label = merge_emotions[emo_label]
               
                #### assign labels for each frame and possibly filter some frames
                frame_indices = [begin_time + ind*frame_len  for ind in range(seg_count_frames)]
                frame_indices = list(map(lambda x:int(x/frame_len), frame_indices))
                new_labels += [emo_label_original  for ind in range(seg_count_frames)]
        new_labels = new_labels[::subsampling_factor]
        if abs(len(utt_labels_fromdataloader) - len(new_labels)) > 2:
            print(f'length of labels from dataloader {len(utt_labels_fromdataloader)} and from data dir {len(new_labels)}, utt_id is {utt_id}')
        new_labels = new_labels[:len(utt_labels_fromdataloader)]
        valid_ind = [i for i,label in enumerate(new_labels) if not label in label_interest]
        temp = [utt_labels_fromdataloader[i] for i in valid_ind]
        new_y_true.append(temp)
        temp = [utt_model_labels[i] for i in valid_ind]
        new_y_pred.append(temp)

    return new_y_true, new_y_pred
    

results_pkl_path_list = sys.argv[1]
collar = int(sys.argv[2])
metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

results_pkl_path_list = results_pkl_path_list.split(',')

print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')

print(f'Calculating diarization metrics for Collar={collar} frames')
y_true_all = []
y_pred_all = []

der_all = []
label_interest = ['overlap']
print(f'labels of interest are {label_interest}')
cv_list = [1, 2, 3, 4, 5]
for cv_ind in cv_list:
    results_pkl_path = results_pkl_path_list[0] + str(cv_ind) + results_pkl_path_list[1]
    data_dir = '/export/b15/rpapagari/Tianzi_work/IEMOCAP_dataset/data_ERC_v2/cv_' + str(cv_ind) + '/'

    print(f'processing {results_pkl_path}')
    results = pkl.load(open(results_pkl_path, 'rb'))    
    target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'
    if os.path.exists(target_label_encoder_path):
        target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
        emo2ind_map = {emo:target_label_encoder.transform([emo])[0] for emo in target_label_encoder.classes_}
        label_list = list(target_label_encoder.classes_)
    else:
        print(f'target_label_encoder doesnt exist')
        label_list = list(set(flatten(y_true)))
        emo2ind_map = {emo:ind for ind,emo in enumerate(sorted(label_list))}
    ind2emo_map = {ind:emo for emo,ind in emo2ind_map.items()}
    #####################################

    temp1, temp2 = filter_results(results['true_labels'], results['predictions'], results['utt_id'], data_dir, label_interest=label_interest)
    results['true_labels'] = temp1
    results['predictions'] = temp2
    diarize_metrics = compute_zhao_kawahara_metrics_speech(results['true_labels'], results['predictions'], results['true_seg_boundaries'], 'Exact', collar=collar, ignore_labels=[])
    der_all.append(diarize_metrics['model_ier']['identification error rate'])
   
    y_true = list(flatten(results['true_labels']))
    y_pred = list(flatten(results['predictions']))
    y_true_all += y_true
    y_pred_all += y_pred

print(f'##########################')
print(f'der is {der_all}')
print(f'ave_der is {np.mean(der_all)}')

labels_list = list(target_label_encoder.classes_)
print(labels_list)
no_results_files = len(cv_list)

calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files)

