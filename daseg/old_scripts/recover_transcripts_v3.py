from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
import pandas as pd



def obtain_truecasing_punctuation_stats(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, punctdescription2punct_map, casing_classes):
    casing_correct = {} 
    casing_error = {}
    casing_total = {}
    
    for punct in punctdescription2punct_map.keys():
        casing_error[punct] = {emo:0 for emo in casing_classes}
        casing_correct[punct] = {emo:0 for emo in casing_classes}
        casing_total[punct] = {emo:0 for emo in casing_classes}
    
    casing_error_word_length = {}
    casing_correct_word_length = {}

    for doc_ind,doc_word2subtokens in enumerate(originalword2subtokens):
        original_doc = ''
        predicted_doc = ''
        #import pdb; pdb.set_trace()
        doc_word2subtokens = doc_word2subtokens[0] 
        for word_ind,word2subtokens in enumerate(doc_word2subtokens):
            true_word = word2subtokens[0]
            try:
                original_doc += true_word + ' '
            except:
                import pdb; pdb.set_trace()
            subtokens = word2subtokens[1]
            subtokens = ''.join(subtokens)
            
            true_punct = y_true1[doc_ind][word_ind]
            pred_punct = y_pred1[doc_ind][word_ind]
    
            true_casing = y_true0[doc_ind][word_ind]
            pred_casing = y_pred0[doc_ind][word_ind]

            #if (true_casing != pred_casing) and (true_casing == 'LC') and (pred_casing == 'AUC'):
            #    print(true_word, subtokens, pred_casing)

            if not len(true_word) in casing_error_word_length:
                casing_error_word_length[len(true_word)] = 0
                casing_correct_word_length[len(true_word)] = 0

            if (true_casing != pred_casing):
                casing_error_word_length[len(true_word)] += 1

            if (true_casing == pred_casing):
                casing_correct_word_length[len(true_word)] += 1

            casing_correct[true_punct][true_casing] += 1*(true_casing == pred_casing)
            casing_error[true_punct][true_casing] += 1*(true_casing != pred_casing)
            casing_total[true_punct][true_casing] += 1

            if pred_casing == 'AUC':
                subtokens = subtokens.upper()
            elif pred_casing == 'UC':
                subtokens = subtokens[0].upper() + subtokens[1:]

            if pred_punct != 'Blank':        
                punct = punctdescription2punct_map[pred_punct]             
                subtokens += punct
            predicted_doc += subtokens + ' '            
    
    print(f'Error statistics according to word length are  ')
    possible_word_lengths = sorted(casing_error_word_length.keys())
    for word_len in possible_word_lengths:
        error_count = casing_error_word_length[word_len]
        correct_count = casing_correct_word_length[word_len]
        total_count = error_count + correct_count + 0.000001
        ratio_of_errors = error_count / total_count
        print(word_len, error_count, correct_count, total_count, ratio_of_errors)
 
    return casing_error, casing_correct, casing_total


## kpy recover_transcripts_v2.py  TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/results.pkl


results_pkl_path = sys.argv[1]
data_dir_tsv = sys.argv[2]
out_result_suffix = sys.argv[3]


punct_label_list = ['', '!', '?', '.', ';', '--', ',', '...']
punct2punctdescription_map = {'':'Blank', '!':'Exclamation', '?':'Question', '.':'FullStop', ';':'SemiColon', '--':'TwoHyphens', ',':'Comma', '...':'Ellipsis'}
punctdescription2punct_map = {j:i for i,j in punct2punctdescription_map.items()}

if results_pkl_path != '-1':
    expt_dir = os.path.dirname(results_pkl_path)
    results = pkl.load(open(results_pkl_path, 'rb'))
    target_label_encoder_path = expt_dir + '/target_label_encoder.pkl'

    if os.path.exists(target_label_encoder_path):
        target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
        
        emo2ind_map0 = {emo:target_label_encoder['0'].transform([emo])[0] for emo in target_label_encoder['0'].classes_}
        emo2ind_map1 = {emo:target_label_encoder['1'].transform([emo])[0] for emo in target_label_encoder['1'].classes_}
        ind2emo_map0 = {ind:emo for emo,ind in emo2ind_map0.items()}
        ind2emo_map1 = {ind:emo for emo,ind in emo2ind_map1.items()}
        casing_classes = sorted(emo2ind_map0.keys())
        print(f'casing_classes are {casing_classes}')
        y_true0 = results['true_labels'+'_op0']
        y_pred0 = results['predictions'+'_op0']
        y_true1 = results['true_labels'+'_op1']
        y_pred1 = results['predictions'+'_op1']
        originalword2subtokens = results['originalword2subtokens']
else:
    y_true0 = []
    y_true1 = []
    originalword2subtokens = []
    expt_dir = os.path.dirname(data_dir_tsv)
    casing_classes = []
    with open(data_dir_tsv, 'r') as f:
        for i in f.readlines():
            doc2label_path = i.split(',')
            doc_labels = pd.read_csv(doc2label_path[1].strip(), sep=',') #header is original_word,word,label,label2        
            doc_labels = doc_labels.dropna()
            doc_labels['subtokens'] = doc_labels['word']
           
            originalword2subtokens += [[[[word2subtokens[0], [word2subtokens[1]]] for word2subtokens in zip(doc_labels['word'], doc_labels['subtokens'])]]]
            #originalword2subtokens += [[doc_labels['word'].values, doc_labels['subtokens'].values]]
            y_true0 += [list(doc_labels['label'].values)]
            y_true1 += [list(doc_labels['label2'].values)]    
    y_pred0 = y_true0
    y_pred1 = y_true1
    casing_classes = sorted(set(flatten(y_true0)))

casing_error, casing_correct, casing_total = obtain_truecasing_punctuation_stats(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, punctdescription2punct_map, casing_classes)


df_total = []
df_error_perc = []
for punct in punctdescription2punct_map.keys():
    print(f'error statistics for words after punctation {punct} are')
    print(casing_error[punct])
    print(casing_correct[punct])
    print(casing_total[punct])
    
    ratio_of_errors = {}

    for casing in casing_classes:
        total_instances = casing_total[punct][casing]
        total_instances += 0.000001
        ratio_of_errors[casing] = casing_error[punct][casing] / total_instances
    print(ratio_of_errors)
    
    temp = [punct] + [ratio_of_errors[casing] for casing in casing_classes]
    df_error_perc += [temp]
    temp = [punct] + [casing_total[punct][casing] for casing in casing_classes]
    df_total += [temp]
    print('\n')


df_total = pd.DataFrame(df_total, columns=['Punct', 'AUC', 'LC', 'UC'])
df_error_perc = pd.DataFrame(df_error_perc, columns=['Punct', 'AUC', 'LC', 'UC'])

if results_pkl_path != '-1':
    df_error_perc.to_csv(expt_dir+'/Casing_Vs_Punctuation_ErrorPerc.csv'+out_result_suffix, sep=',', index=False)
df_total.to_csv(expt_dir+'/Casing_Vs_Punctuation_Dataset.csv'+out_result_suffix, sep=',', index=False)



