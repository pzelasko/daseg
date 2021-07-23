from more_itertools import flatten
import sklearn.metrics as sklmetrics
from daseg.metrics_v2 import compute_sklearn_metrics, compute_zhao_kawahara_metrics_speech, conv_level_emo_pred, conv_level_senti_pred
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate


def calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files, emo2ind_map):
    conf_mat = sklmetrics.confusion_matrix(y_true_all, y_pred_all, labels=labels_list)
    conf_mat_original = deepcopy(conf_mat)
    conf_mat = conf_mat/no_results_files
    conf_mat = conf_mat.astype('int')
    classwise_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average=None)
    micro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='weighted')
    macro_f1 = sklmetrics.f1_score(y_true_all, y_pred_all, average='macro')
    unweighted_acc = sklmetrics.accuracy_score(y_true_all, y_pred_all)
    classif_report = sklmetrics.classification_report(y_true_all, y_pred_all, labels=labels_list)

    support = np.sum(conf_mat, axis=-1)
    class_wts = support/np.sum(support)
    y_true_all_ind = np.array([emo2ind_map[emo] for emo in y_true_all])
    y_pred_all_ind = np.array([emo2ind_map[emo] for emo in y_pred_all])
    no_samples = len(y_true_all_ind)
    sample_wts = np.ones(no_samples) 
    ## obtain weight for each sample in the data
    for idx, i in enumerate(np.bincount(y_true_all_ind)): 
        sample_wts[y_true_all_ind==idx] *= i/float(no_samples)
    weighted_acc = sklmetrics.accuracy_score(y_true_all_ind, y_pred_all_ind, sample_weight=sample_wts)

    #print(classwise_acc)
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
            #print(f'classwise acc: {classwise_acc}')
            print(f'weighted_acc: {weighted_acc}')            

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
seg_eval = 'false' # sys.argv[3]

metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

seg_eval = (lambda x:x.lower()=='true')(seg_eval)


results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

print(results_pkl_path_list)
print(f'no. of results files are {len(results_pkl_path_list)}')

emo2sentiment_map = {'ang':'neg', 'fru':'neg', 'hap':'pos', 'neu':'pos', 'sad':'neg'}

print(f'Calculating diarization metrics for Collar={collar} frames')
y_true_all = []
y_pred_all = []
y_true_conv_all = []
y_pred_conv_all = []
y_true_conv_senti_all = []
y_pred_conv_senti_all = []

der_all = []
der_all_ignore_labels = []
ignore_labels = ['silence'] #['OOS', 'silence'] #['OOS'] #, 'silence']
for results_ind,results_pkl_path in enumerate(results_pkl_path_list):
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
    
    #diarize_metrics = compute_zhao_kawahara_metrics_speech(results['true_labels'], results['predictions'], results['true_seg_boundaries'], 'Exact', collar=collar, ignore_labels=ignore_labels)
    #der_all.append(diarize_metrics['model_ier']['identification error rate'])
    #der_all_ignore_labels.append(diarize_metrics['model_ier_OnlyMainEmo']['identification error rate'])
    
    #try:
    #### calculating conversational level metrics #########
    #    conv_true_pred, conv_model_pred, conv_level_emo_acc = conv_level_emo_pred(results['logits'], results['true_labels'], emo2ind_map, ind2emo_map, ignore_neu=False) 
    #    conv_true_senti_pred, conv_model_senti_pred, conv_level_senti_acc = conv_level_senti_pred(conv_true_pred, conv_model_pred, emo2sentiment_map)
    #    y_true_conv_all += conv_true_pred
    #    y_pred_conv_all += conv_model_pred
    #    y_true_conv_senti_all += conv_true_senti_pred
    #    y_pred_conv_senti_all += conv_model_senti_pred
    #except: 
    #    pass
    
    #import pdb; pdb.set_trace()
    y_true = list(flatten(results['true_labels']))
    y_pred = list(flatten(results['predictions']))
    #if 'zhao_kawahara_metrics' in results:
    #    print(results['zhao_kawahara_metrics'])
    y_true_all += y_true
    y_pred_all += y_pred


############################ metrics per segment duration bucket  ############
if seg_eval:
    try:
        min_seg_dur = [0, 1, 2, 3, 5, 10]
        max_seg_dur = [1, 2, 3, 5, 10, 100]
        calc_acc_per_seg_dur(y_true_all, y_pred_all, frame_len=0.08, min_seg_dur=min_seg_dur, max_seg_dur=max_seg_dur)
        
        #min_seg_dur = [0]
        #max_seg_dur = [100]
        #calc_acc_per_seg_dur_surroundemotion(y_true_all, y_pred_all, frame_len=0.08, min_seg_dur=min_seg_dur, max_seg_dur=max_seg_dur)
    except:
        pass
###########################


print(f'##########################')
print(f'der is {der_all}')
print(f'ave_der is {np.mean(der_all)}')

#labels_list = sorted(set(y_true_all))
labels_list = list(target_label_encoder.classes_)
print(labels_list)
no_results_files = len(results_pkl_path_list)

calc_overall_metrics(y_true_all, y_pred_all, labels_list, no_results_files, emo2ind_map)

