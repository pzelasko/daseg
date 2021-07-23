from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate


results_pkl_path = sys.argv[1]

results = pkl.load(open(results_pkl_path, 'rb'))
target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'


punct_label_list = ['', '!', '?', '.', ';', '--', ',', '...']
punct2punctdescription_map = {'':'Blank', '!':'Exclamation', '?':'Question', '.':'FullStop', ';':'SemiColon', '--':'TwoHyphens', ',':'Comma', '...':'Ellipsis'}
punctdescription2punct_map = {j:i for i,j in punct2punctdescription_map.items()}


if os.path.exists(target_label_encoder_path):
    target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
    
    emo2ind_map0 = {emo:target_label_encoder['0'].transform([emo])[0] for emo in target_label_encoder['0'].classes_}
    emo2ind_map1 = {emo:target_label_encoder['1'].transform([emo])[0] for emo in target_label_encoder['1'].classes_}
    ind2emo_map0 = {ind:emo for emo,ind in emo2ind_map0.items()}
    ind2emo_map1 = {ind:emo for emo,ind in emo2ind_map1.items()}


    y_true0 = results['true_labels'+'_op0']
    y_pred0 = results['predictions'+'_op0']

    y_true1 = results['true_labels'+'_op1']
    y_pred1 = results['predictions'+'_op1']

    originalword2subtokens = results['originalword2subtokens']
    original_doc = ''
    predicted_doc = ''
    for doc_ind,doc_word2subtokens in enumerate(originalword2subtokens):
        import pdb; pdb.set_trace()
        doc_word2subtokens = doc_word2subtokens[0] 
        for word_ind,word2subtokens in enumerate(doc_word2subtokens):
            original_doc += word2subtokens[0] + ' '
            subtokens = word2subtokens[1]
            subtokens = ''.join(subtokens)
            
            true_punct = y_true1[doc_ind][word_ind]
            pred_punct = y_pred1[doc_ind][word_ind]
    
            true_casing = y_true0[doc_ind][word_ind]
            pred_casing = y_pred0[doc_ind][word_ind]

            if (true_casing != pred_casing) and (true_casing == 'LC') and (pred_casing == 'AUC'):
                print(word2subtokens[0], subtokens, pred_casing)

            if pred_casing == 'AUC':
                subtokens = subtokens.upper()
            elif pred_casing == 'UC':
                subtokens = subtokens[0].upper() + subtokens[1:]

            if pred_punct != 'Blank':        
                punct = punctdescription2punct_map[pred_punct]             
                subtokens += punct
            predicted_doc += subtokens + ' '            
    
    

