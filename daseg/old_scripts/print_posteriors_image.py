import os, sys
import pickle as pkl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np


############ obtaining conversational level labels, both ground truth and model predictions  #################
def conv_level_emo_pred(logits, ind2emo_map, y_true):
    logits = torch.tensor(logits)
    model_posteriors = nn.functional.softmax(logits, dim=-1)
    model_posteriors = model_posteriors.to('cpu').numpy()

    conv_model_pred = np.sum(model_posteriors, axis=1) # sum along time dimension
    if ignore_neu:
        neu_ind = emo2ind_map['neu']
        conv_model_pred[:, neu_ind] = 0

    conv_model_pred = np.argmax(conv_model_pred, axis=-1)
    conv_model_pred = [ind2emo_map[i] for i in conv_model_pred]
    
    conv_true_pred = []
    for utt_ind in range(len(y_true)):
        utt_labels = y_true[utt_ind]
        if ignore_neu:
            utt_labels = [i for i in utt_labels if i!='neu']

        most_freq_emo = max(set(utt_labels), key=utt_labels.count)
        conv_true_pred.append(most_freq_emo)
    
    print(list(zip(conv_true_pred, conv_model_pred)))
    conv_level_emo_acc = np.mean([i==j for i,j in zip(conv_true_pred, conv_model_pred)])
    print(f'conv_level_emo_acc is {conv_level_emo_acc}')
    return conv_true_pred, conv_model_pred, conv_level_emo_acc


def conv_level_senti_pred(conv_true_pred, conv_model_pred, emo2sentiment_map):
    conv_true_senti_pred = [emo2sentiment_map[i] for i in conv_true_pred]
    conv_model_senti_pred = [emo2sentiment_map[i] for i in conv_model_pred]
    conv_level_senti_acc = np.mean([i==j for i,j in zip(conv_true_senti_pred, conv_model_senti_pred)])    
    print(list(zip(conv_true_senti_pred, conv_model_senti_pred)))
    print(f'conv level senti acc is {conv_level_senti_acc}')


results_pkl_path = 'journal_v2/IEMOCAP_v2_CV_5_ExactLabelScheme_smoothSegmentation_SoTAEnglishXVectorTrue_ASHNF_ExciteMapHap_clean_noise123_music123_frame_len0.01_ConvClassif_bs3_gacc10_concat_aug_0_warmup200steps_xformer_maxseqlen_2048/xformer_IEMOCAP_v2_42/results.pkl'

ignore_neu = True

results = pkl.load(open(results_pkl_path, 'rb'))
y_true = results['true_labels']
y_pred = results['predictions']


emo2sentiment_map = {'ang':'neg', 'fru':'neg', 'hap':'pos', 'neu':'neu', 'sad':'neg'}
emo2sentiment_map = {'ang':'neg', 'fru':'neg', 'hap':'pos', 'neu':'pos', 'sad':'neg'}


conv_true_pred, conv_model_pred, conv_level_emo_acc = conv_level_emo_pred(logits, ind2emo_map, y_true)
conv_level_senti_pred(conv_true_pred, conv_model_pred, emo2sentiment_map)



