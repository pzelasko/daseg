from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
import pandas as pd


def recover_word_with_modelpreds(model_ip_word, pred_casing, pred_punct):
    predicted_word = model_ip_word
    if INCLUDE_PRED_CASING:
        if pred_casing == 'AUC':
            predicted_word = predicted_word.upper()
        elif pred_casing == 'UC':
            predicted_word = predicted_word[0].upper() + predicted_word[1:]

    if (INCLUDE_PRED_PUNCT) and (pred_punct != 'Blank'):
        punct = PUNCTDESCRIPTION2PUNCT_MAP[pred_punct]             
        predicted_word += punct
    return predicted_word


def obtain_truecasing_punctuation_stats(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, out_data_dir, split):
    casing_correct = {} 
    casing_error = {}
    casing_total = {}
    
    for punct in PUNCTDESCRIPTION2PUNCT_MAP.keys():
        casing_error[punct] = {emo:0 for emo in CASING_CLASSES}
        casing_correct[punct] = {emo:0 for emo in CASING_CLASSES}
        casing_total[punct] = {emo:0 for emo in CASING_CLASSES}
    
    casing_error_word_length = {}
    casing_correct_word_length = {}

    predicted_doc_all = []
    original_doc_all = []
    model_ip_doc_all = []

    for doc_ind,doc_word2subtokens in enumerate(originalword2subtokens):
        original_doc = ''
        predicted_doc = ''
        model_ip_doc = ''
        #import pdb; pdb.set_trace()
        doc_word2subtokens = doc_word2subtokens[0] 
        word2truecasing_punctlabels = []
        for word_ind in range(len(doc_word2subtokens)-1):
            word2subtokens = doc_word2subtokens[word_ind]
            true_word = word2subtokens[0]
            try:
                original_doc += true_word + ' '
            except:
                import pdb; pdb.set_trace()
            subtokens = word2subtokens[1]
            subtokens = ''.join(subtokens)
            
            true_punct = y_true1[doc_ind][word_ind]
            pred_punct = y_pred1[doc_ind][word_ind]
    
            ### need to use "word_ind+1" instead of "word_ind" to get the relation between truecasing and punctuation 
            true_casing = y_true0[doc_ind][word_ind+1]
            pred_casing = y_pred0[doc_ind][word_ind+1]
            
            ## get the original model input word (combining subtokens do not give input word sometimes
            ## Ex: tokenize('Ebony') --> 'e', '##bon', '##y' --> extra hashes are inserted
             
            model_ip_word = true_word.lower()
            if true_punct != 'Blank':
                model_ip_word = model_ip_word[:-len(PUNCTDESCRIPTION2PUNCT_MAP[true_punct])]
            model_ip_doc += model_ip_word + ' '

            ## once you have model_ip_word (word given to model), you can recover casing and punctaution from model predictions in the following manner
            predicted_word = recover_word_with_modelpreds(model_ip_word, pred_casing, pred_punct)
            predicted_doc += predicted_word + ' '            
    
            ### for error statistics
            casing_correct[true_punct][true_casing] += 1*(true_casing == pred_casing)
            casing_error[true_punct][true_casing] += 1*(true_casing != pred_casing)
            casing_total[true_punct][true_casing] += 1

        original_doc_all.append(original_doc)
        predicted_doc_all.append(predicted_doc)
        model_ip_doc_all.append(model_ip_doc)
 
    return casing_error, casing_correct, casing_total, original_doc_all, predicted_doc_all, model_ip_doc_all


def recover_truecasing_punctuation(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, out_data_dir, split):

    predicted_doc_all = []
    original_doc_all = []
    model_ip_doc_all = []

    # define transcripts_op_dir, utt2csvpath_f, data_tsv_f
    data_tsv_path = out_data_dir + '/' + split + '.tsv'
    data_tsv_f = open(data_tsv_path, 'w')
    transcripts_op_dir = out_data_dir + '/transcripts_withlabels/' + split + '/'
    utt2csvpath_f = open(out_data_dir + '/utt2csvpath_' + split, 'w')
    
    os.makedirs(transcripts_op_dir, exist_ok=True)

    for doc_ind,doc_word2subtokens in enumerate(originalword2subtokens):
        original_doc = ''
        predicted_doc = ''
        model_ip_doc = ''
        #import pdb; pdb.set_trace()
        doc_word2subtokens = doc_word2subtokens[0] 
        word2truecasing_punctlabels = []
        doc_id = split + '_doc_' + str(doc_ind)
        op_doc_path = transcripts_op_dir + '/' + doc_id + '.txt'

        for word_ind in range(len(doc_word2subtokens)):
            word2subtokens = doc_word2subtokens[word_ind]
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
            
            ## get the original model input word (combining subtokens do not give input word sometimes
            ## Ex: tokenize('Ebony') --> 'e', '##bon', '##y' --> extra hashes are inserted
             
            model_ip_word = true_word.lower()
            if true_punct != 'Blank':
                model_ip_word = model_ip_word[:-len(PUNCTDESCRIPTION2PUNCT_MAP[true_punct])]
            model_ip_doc += model_ip_word + ' '

            ## once you have model_ip_word (word given to model), you can recover casing and punctaution from model predictions in the following manner
            predicted_word = recover_word_with_modelpreds(model_ip_word, pred_casing, pred_punct)
            predicted_doc += predicted_word + ' '            
    

            word2truecasing_punctlabels += [[true_word, predicted_word, true_casing, true_punct]]
            #if (INCLUDE_PRED_PUNCT) and INCLUDE_PRED_CASING:
            #    word2truecasing_punctlabels += [[true_word, predicted_word, true_casing, true_punct]]
            #elif INCLUDE_PRED_CASING:
            #    # if casing predictions are included into the input words then we can only use it for punctuation models training
            #    word2truecasing_punctlabels += [[true_word, predicted_word, 'DoNotExist', true_punct]]
            #elif INCLUDE_PRED_PUNCT:
            #    ## if punctuation predictions are included into the input words then we can only use it for Truecasing models training
            #    word2truecasing_punctlabels += [[true_word, predicted_word, true_casing, 'DoNotExist']]
    

        #import pdb; pdb.set_trace()
        
        word2labels_df = pd.DataFrame(word2truecasing_punctlabels, columns=['original_word', 'word', 'label', 'label2'])
        word2labels_df.to_csv(op_doc_path, sep=',', index=False)

        utt2csvpath_f.write(doc_id + ',' + op_doc_path + '\n')
        data_tsv_f.write(doc_id + ',' + op_doc_path + '\n')

        original_doc_all.append(original_doc)
        predicted_doc_all.append(predicted_doc)
        model_ip_doc_all.append(model_ip_doc)
 
    return original_doc_all, predicted_doc_all, model_ip_doc_all


def prepare_pickle(predicted_doc_all):
    doc_all = []
    for doc_ind,doc in enumerate(predicted_doc_all):
        turn = {}
        _, turn['words'] = doc
        docs_all.append([turn])
    return doc_all
    


## kpy recover_transcripts_v2.py  TrueCasing_expts_data_v3/TrueCasing_CV_1_ExactLabelScheme_smoothSegmentation_text_model_truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_WOTurnToken_CombineUCandMC_8classes_loss_wts_1_0_Epochs_25/truecasing_punctuation_Morethan2TasksArch_longformer_tokenclassif_TrueCasing_42/results.pkl


results_pkl_path = sys.argv[1] # -1 if you are calculating stats from dataset dir instead of results dir
data_dir_tsv = sys.argv[2] # not needed if calculating from result dir
out_result_suffix = sys.argv[3] # used to name output files from this code
data_split = sys.argv[4]
out_data_dir = sys.argv[5]
INCLUDE_PRED_PUNCT = sys.argv[6]
INCLUDE_PRED_CASING = sys.argv[7]

## INCLUDE_PRED_PUNCT and INCLUDE_PRED_CASING are True if you want to recover both. Usually used for analysis


INCLUDE_PRED_PUNCT = (lambda x:x.lower()=='true')(INCLUDE_PRED_PUNCT)
INCLUDE_PRED_CASING = (lambda x:x.lower()=='true')(INCLUDE_PRED_CASING)

print(f'INCLUDE_PRED_PUNCT is {INCLUDE_PRED_PUNCT}')
print(f'INCLUDE_PRED_CASING is {INCLUDE_PRED_CASING}')

os.makedirs(out_data_dir, exist_ok=True)

punct_label_list = ['', '!', '?', '.', ';', '--', ',', '...']
PUNCT2PUNCTDESCRIPTION_MAP = {'':'Blank', '!':'Exclamation', '?':'Question', '.':'FullStop', ';':'SemiColon', '--':'TwoHyphens', ',':'Comma', '...':'Ellipsis'}
PUNCTDESCRIPTION2PUNCT_MAP = {j:i for i,j in PUNCT2PUNCTDESCRIPTION_MAP.items()}

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
        CASING_CLASSES = sorted(emo2ind_map0.keys())
        print(f'CASING_CLASSES are {CASING_CLASSES}')
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
    CASING_CLASSES = []
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
    CASING_CLASSES = sorted(set(flatten(y_true0)))

## used to get TC Vs. Punctuation stats
casing_error, casing_correct, casing_total, original_doc_all, predicted_doc_all, model_ip_doc_all = obtain_truecasing_punctuation_stats(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, out_data_dir, data_split)

## used to recover transcripts
#original_doc_all, predicted_doc_all, model_ip_doc_all = recover_truecasing_punctuation(y_true0, y_pred0, y_true1, y_pred1, originalword2subtokens, out_data_dir, data_split)
#sys.exit()


df_total = []
df_error_perc = []
for punct in PUNCTDESCRIPTION2PUNCT_MAP.keys():
    print(f'error statistics for words after punctation {punct} are')
    print(casing_error[punct])
    print(casing_correct[punct])
    print(casing_total[punct])
    
    ratio_of_errors = {}

    for casing in CASING_CLASSES:
        total_instances = casing_total[punct][casing]
        total_instances += 0.000001
        ratio_of_errors[casing] = casing_error[punct][casing] / total_instances
    print(ratio_of_errors)
    
    temp = [punct] + [ratio_of_errors[casing] for casing in CASING_CLASSES]
    df_error_perc += [temp]
    temp = [punct] + [casing_total[punct][casing] for casing in CASING_CLASSES]
    df_total += [temp]
    print('\n')


df_total = pd.DataFrame(df_total, columns=['Punct', 'AUC', 'LC', 'UC'])
df_error_perc = pd.DataFrame(df_error_perc, columns=['Punct', 'AUC', 'LC', 'UC'])

if results_pkl_path != '-1':
    df_error_perc.to_csv(expt_dir+'/Casing_Vs_Punctuation_ErrorPerc.csv'+out_result_suffix, sep=',', index=False)
df_total.to_csv(expt_dir+'/Casing_Vs_Punctuation_Dataset.csv'+out_result_suffix, sep=',', index=False)

df_total = df_total.set_index('Punct')
df_total_norm = df_total.div(df_total.sum(axis=1), axis=0)

df_total_norm.to_csv(expt_dir+'/Casing_Vs_Punctuation_Dataset_Normalized.csv'+out_result_suffix, sep=',', index=False)


