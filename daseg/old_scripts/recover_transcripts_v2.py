from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
import pandas as pd


## kpy recover_transcripts_v2.py  TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/results.pkl


results_pkl_path = sys.argv[1]

expt_dir = os.path.dirname(results_pkl_path)
results = pkl.load(open(results_pkl_path, 'rb'))
target_label_encoder_path = expt_dir + '/target_label_encoder.pkl'


punct_label_list = ['', '!', '?', '.', ';', '--', ',', '...']
punct2punctdescription_map = {'':'Blank', '!':'Exclamation', '?':'Question', '.':'FullStop', ';':'SemiColon', '--':'TwoHyphens', ',':'Comma', '...':'Ellipsis'}
punctdescription2punct_map = {j:i for i,j in punct2punctdescription_map.items()}


casing_correct = {} 
casing_error = {}
casing_total = {}

for punct in punctdescription2punct_map.keys():
    casing_error[punct] = {'AUC':0, 'LC':0, 'UC':0}
    casing_correct[punct] = {'AUC':0, 'LC':0, 'UC':0}
    casing_total[punct] = {'AUC':0, 'LC':0, 'UC':0}

casing_error_word_length = {}
casing_correct_word_length = {}
 
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
    original_doc = ''
    predicted_doc = ''
    for doc_ind,doc_word2subtokens in enumerate(originalword2subtokens):
        #import pdb; pdb.set_trace()
        doc_word2subtokens = doc_word2subtokens[0] 
        for word_ind,word2subtokens in enumerate(doc_word2subtokens):
            true_word = word2subtokens[0]
            original_doc += true_word + ' '
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


            #if (true_punct != 'Blank'): # and (true_punct == pred_punct):
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
    
import pdb; pdb.set_trace()

print(f'Error statistics according to word length are  ')
possible_word_lengths = sorted(casing_error_word_length.keys())
for word_len in possible_word_lengths:
    error_count = casing_error_word_length[word_len]
    correct_count = casing_correct_word_length[word_len]
    total_count = error_count + correct_count + 0.000001
    ratio_of_errors = error_count / total_count
    print(word_len, error_count, correct_count, total_count, ratio_of_errors)
    

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

df_error_perc.to_csv(expt_dir+'/Casing_Vs_Punctuation_ErrorPerc.csv', sep=',', index=False)
df_total.to_csv(expt_dir+'/Casing_Vs_Punctuation_Dataset.csv', sep=',', index=False)



