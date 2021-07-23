from more_itertools import flatten
import sklearn.metrics as sklmetrics
from daseg.metrics_v2 import compute_sklearn_metrics, compute_zhao_kawahara_metrics_speech, conv_level_emo_pred, conv_level_senti_pred
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
from sklearn.metrics import f1_score,  log_loss, accuracy_score, confusion_matrix, classification_report, recall_score, mean_squared_error


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
    
    #print(metrics_list)
    metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2]
    metrics_list = metrics_list.split(',')
    for metric in metrics_list:
        
        if metric == 'confusion':
            print('per result file conf mat is')
            print(tabulate.tabulate(conf_mat, tablefmt='plain'))
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


def calc_trans_prob(y_true_all, y_pred_all):
    trans_prob = {}
    for ind in range(len(y_true_all)):
        if ind > 0:
            pass


def calc_acc_per_seg_dur(y_true_all, y_pred_all, frame_len=0.08, min_seg_dur=0, max_seg_dur=3):
    count = 1
    y_true_seg_dur = []
    for ind,emo in enumerate(y_true_all):
        if ind > 0:
            if emo == y_true_all[ind-1]:
                count += 1
            else:
                seg_dur = count*frame_len
                y_true_seg_dur += [seg_dur for _ in range(count)]
                count = 1
    seg_dur = count*frame_len
    y_true_seg_dur += [seg_dur for _ in range(count)]    

    print(len(y_true_seg_dur), len(y_true_all))
    assert len(y_true_seg_dur) == len(y_true_all)
    labels_list = sorted(list(set(y_true_all)))
    no_results_files = 1
    for min_seg_dur, max_seg_dur in zip(min_seg_dur, max_seg_dur):
        print(f'min_seg_dur, max_seg_dur are {min_seg_dur}, {max_seg_dur}')           
        y_true_interest = []
        y_pred_interest = []    
        for ind,seg_dur in enumerate(y_true_seg_dur):
            if (seg_dur < max_seg_dur) and (seg_dur > min_seg_dur):
                y_true_interest.append(y_true_all[ind])
                y_pred_interest.append(y_pred_all[ind])
        calc_overall_metrics(y_true_interest, y_pred_interest, labels_list, no_results_files)    
    

def calc_acc_per_seg_dur_surroundemotion(y_true_all, y_pred_all, frame_len=0.08, min_seg_dur=0, max_seg_dur=3):
    count = 1
    y_true_seg_dur = []
    y_true_surround_emo1 = []
    y_true_surround_emo2 = []
    for ind,emo in enumerate(y_true_all):
        if ind > 0:
            if emo == y_true_all[ind-1]:
                count += 1
            else:
                seg_dur = count*frame_len
                y_true_seg_dur += [seg_dur for _ in range(count)]
                count = 1
    seg_dur = count*frame_len
    y_true_seg_dur += [seg_dur for _ in range(count)]    

    print(len(y_true_seg_dur), len(y_true_all))
    assert len(y_true_seg_dur) == len(y_true_all)
    labels_list = sorted(list(set(y_true_all)))
    no_results_files = 1
    for min_seg_dur, max_seg_dur in zip(min_seg_dur, max_seg_dur):
        print(f'min_seg_dur, max_seg_dur are {min_seg_dur}, {max_seg_dur}')           
        y_true_interest = []
        y_pred_interest = []    
        y_true_surround_interest = []
        for ind,seg_dur in enumerate(y_true_seg_dur):
            if (seg_dur < max_seg_dur) and (seg_dur > min_seg_dur):
                if y_true_all[ind] == 'neu':
                    #y_true_interest.append(y_true_all[ind])
                    surround_emo = []
                    ## get emo of next segment
                    new_ind = ind+1
                    next_seg_dur = y_true_seg_dur[new_ind]
                    while seg_dur == next_seg_dur:
                        new_ind += 1
                        next_seg_dur = y_true_seg_dur[new_ind]
                    surround_emo.append(y_true_all[new_ind])
                    ## get emo of prev segment
                    try:
                        new_ind = ind-1
                        prev_seg_dur = y_true_seg_dur[new_ind]
                        while seg_dur == prev_seg_dur:
                            new_ind -= 1
                            prev_seg_dur = y_true_seg_dur[new_ind]
                        surround_emo.append(y_true_all[new_ind])    
                    except:
                        pass
                    if (len(surround_emo) == 2) and (surround_emo[0] == surround_emo[1]):
                        y_true_surround_interest.append(surround_emo[0])                
                        y_true_interest.append(y_true_all[ind])
                        y_pred_interest.append(y_pred_all[ind])
                    #else:
                    #    y_true_surround_interest.append(y_true_all[ind])

                    #import pdb; pdb.set_trace() 
                    #y_pred_interest.append(y_pred_all[ind])
        calc_overall_metrics(y_true_interest, y_pred_interest, labels_list, no_results_files)    
        import pdb; pdb.set_trace() 
        calc_overall_metrics(y_true_surround_interest, y_pred_interest, labels_list, no_results_files)


results_pkl_path_list = sys.argv[1]
collar = int(sys.argv[2])
metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')


print(f'Calculating diarization metrics for Collar={collar} frames')
y_true_all = []
y_pred_all = []

for results_ind,results_pkl_path in enumerate(results_pkl_path_list):
    print(f'processing {results_pkl_path}')
    results = pkl.load(open(results_pkl_path, 'rb'))    
    #import pdb; pdb.set_trace()
    y_true = list(flatten(results['true_labels']))
    y_pred = list(flatten(results['predictions']))

    y_true_all += y_true
    y_pred_all += y_pred

no_results_files = len(results_pkl_path_list)


if len(y_true_all) > 0:
    y_pred_all = list(np.vstack(y_pred_all))
    y_true_all = list(np.vstack(y_true_all))
    for i in range(len(y_pred_all)):
        print(y_true_all[i], y_pred_all[i])
    
   
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse*30*30) # for modeling purposes, data dirs have actual_value/30, hence multiplying with 30^2    
    
    print(f'RMSE is {rmse}')
    
     
    #dynamic_range_fun = lambda x:5.5*np.log(100*x) # inverse of this function is used for to transform original MMSE values. hence y_true_all after inverse transformation should be almost equal to original MMSE values
    dynamic_range_fun = lambda x:4*np.log(100*x) # inverse of this function is used for to transform original MMSE values. hence y_true_all after inverse transformation should be almost equal to original MMSE values
    #dynamic_range_fun = lambda x:5.5*np.log(x) # inverse of this function is used for to transform original MMSE values. hence y_true_all after inverse transformation should be almost equal to original MMSE values
    y_pred_all = [dynamic_range_fun(i) for i in y_pred_all]
    y_true_all = [dynamic_range_fun(i) for i in y_true_all]
 
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse) # for modeling purposes, data dirs have actual_value/30, hence multiplying with 30^2

    print(f'RMSE when dynamic_range_fun is used -- {rmse}')


