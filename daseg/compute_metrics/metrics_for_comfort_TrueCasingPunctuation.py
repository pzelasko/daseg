from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
#from daseg.metrics_v2 import compute_sklearn_metrics, compute_zhao_kawahara_metrics_speech, conv_level_emo_pred, conv_level_senti_pred


def calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files):
    conf_mat = sklmetrics.confusion_matrix(y_true_all, y_pred_all, labels=labels_list)
    conf_mat_original = deepcopy(conf_mat)
    conf_mat = conf_mat/no_results_files
    conf_mat = conf_mat.astype('int')
    classwise_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average=None)
    micro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='weighted')
    macro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='macro')
    unweighted_acc = sklmetrics.accuracy_score(y_true_all, y_pred_all)
    classif_report = sklmetrics.classification_report(y_true_all, y_pred_all, labels=labels_list)
    
    #import pdb; pdb.set_trace()
    #print(metrics_list)
    metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2]
    metrics_list = metrics_list.split(',')
    for metric in metrics_list:
        
        if metric == 'confusion':
            print('per result file conf mat is')
            print(tabulate.tabulate(conf_mat, tablefmt='plain'))
            print(labels_list)
            print('original conf mat is ')
            print(tabulate.tabulate(conf_mat_original, tablefmt='plain'))
            print(labels_list)
        if metric == 'f1-score':
            print(f'classwise_f1: {classwise_f1}')
            print(f'micro_f1: {micro_f1}')
            print(f'macro_f1: {macro_f1}')
        if metric == 'accuracy':
            print(f'unweighted_acc: {unweighted_acc}')                    
    print(classif_report)


results_pkl_path_list = sys.argv[1]
label_block_name = sys.argv[2]

metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')

y_true_all = []
y_pred_all = []
for results_ind,results_pkl_path in enumerate(results_pkl_path_list):
    print(f'processing {results_pkl_path}')
    results = pkl.load(open(results_pkl_path, 'rb'))    
    target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'
    if os.path.exists(target_label_encoder_path):
        target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
        if label_block_name != '-1':
            target_label_encoder = target_label_encoder[label_block_name]

        emo2ind_map = {emo:target_label_encoder.transform([emo])[0] for emo in target_label_encoder.classes_}
        label_list = list(target_label_encoder.classes_)
    else:
        print(f'target_label_encoder doesnt exist')
        label_list = list(set(flatten(y_true)))
        emo2ind_map = {emo:ind for ind,emo in enumerate(sorted(label_list))}
    ind2emo_map = {ind:emo for emo,ind in emo2ind_map.items()}
    #####################################
    
    #import pdb; pdb.set_trace()

    if label_block_name != '-1':
        true_labels_key = [i for i in results.keys() if i.startswith('true_labels') if i.endswith('_op'+label_block_name)]
        predictions_key = [i for i in results.keys() if i.startswith('predictions') if i.endswith('_op'+label_block_name)]

        true_labels_key = true_labels_key[0]
        predictions_key = predictions_key[0]
    else:
        true_labels_key = 'true_labels'
        predictions_key = 'predictions'
    print(true_labels_key, predictions_key)
    
    y_true = list(flatten(results[true_labels_key]))
    y_pred = list(flatten(results[predictions_key]))
    y_true_all += y_true
    y_pred_all += y_pred


labels_list = list(target_label_encoder.classes_)
print(labels_list)
no_results_files = len(results_pkl_path_list)

calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files)

