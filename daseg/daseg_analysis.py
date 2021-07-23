from more_itertools import flatten
import sklearn.metrics as sklmetrics
from daseg.metrics_v2 import compute_sklearn_metrics, compute_zhao_kawahara_metrics_speech, compute_zhao_kawahara_metrics_DASEG_analysis
import pickle as pkl
import os, sys
import glob
import pyannote.metrics #.detection.DetectionPrecisionRecallFMeasure as 
import numpy as np


results_pkl_path_list = sys.argv[1] #/export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_lower_basic_42/results.pkl
collar = int(sys.argv[2])
metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 


results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')
metrics_list = metrics_list.split(',')

print(f'Calculating diarization metrics for Collar={collar} frames')
y_true_all = []
y_pred_all = []
der_all = []
for results_ind,results_pkl_path in enumerate(results_pkl_path_list):
    results = pkl.load(open(results_pkl_path, 'rb'))    

    
    target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'
    if os.path.isfile(target_label_encoder_path):
        target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))    
        labels_list = list(target_label_encoder.classes_)
    else:
        labels_list = list(set(flatten(results['true_labels'])))

    label_scheme = 'E'
    diarize_metrics = compute_zhao_kawahara_metrics_DASEG_analysis(results['true_labels'], results['predictions'], 'temp', label_scheme, collar=collar)
    import pdb; pdb.set_trace()
    der_all.append(diarize_metrics['model_ier']['identification error rate'])
    
    y_true = list(flatten(results['true_labels']))
    y_pred = list(flatten(results['predictions']))
    if 'zhao_kawahara_metrics' in results:
        print(results['zhao_kawahara_metrics'])
    y_true_all += y_true
    y_pred_all += y_pred

print(f'##########################')
print(f'der is {der_all}')
print(f'ave_der is {np.mean(der_all)}')

if not os.isfile(target_label_encoder_path):
    labels_list = sorted(set(y_true_all))
else:
    labels_list = list(target_label_encoder.classes_)
print(labels_list)

conf_mat = sklmetrics.confusion_matrix(y_true_all, y_pred_all, labels=labels_list)
conf_mat = conf_mat/len(results_pkl_path_list)
conf_mat = conf_mat.astype('int')
classwise_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average=None)
micro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='weighted')
macro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='macro')
unweighted_acc = sklmetrics.accuracy_score(y_true_all, y_pred_all)
classif_report = sklmetrics.classification_report(y_true_all, y_pred_all, labels=labels_list)

print(metrics_list)
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

