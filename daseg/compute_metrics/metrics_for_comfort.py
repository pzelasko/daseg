from more_itertools import flatten
import sklearn.metrics as sklmetrics
from daseg.metrics import compute_sklearn_metrics
import pickle as pkl
import os, sys
import glob


results_pkl_path_list = sys.argv[1]
metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

#print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')
metrics_list = metrics_list.split(',')

y_true_all = []
y_pred_all = []
for results_pkl_path in results_pkl_path_list:
    results = pkl.load(open(results_pkl_path, 'rb'))    
    target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'
    target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
    
    labels_list = list(target_label_encoder.classes_)

    y_true = list(flatten(results['true_labels']))
    y_pred = list(flatten(results['predictions']))
    if 'zhao_kawahara_metrics' in results:
        print(results['zhao_kawahara_metrics'])
    y_true_all += y_true
    y_pred_all += y_pred

labels_list = sorted(set(y_true_all))
print(labels_list)

conf_mat = sklmetrics.confusion_matrix(y_true_all, y_pred_all, labels=labels_list)
conf_mat = conf_mat/len(results_pkl_path_list)
conf_mat = conf_mat.astype('int')
classwise_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average=None)
micro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='weighted')
macro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='macro')
unweighted_acc = sklmetrics.accuracy_score(y_true_all, y_pred_all)

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
        



