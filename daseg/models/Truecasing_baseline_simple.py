import sys, os
import pandas as pd
import numpy as np
import logging, tabulate


def calc_weighted_acc_from_conf_mat(conf_mat, weights):
    den = np.sum(conf_mat, axis=-1)*weights
    num = np.diag(conf_mat)*weights
    return sum(num)/sum(den)


def convert_to_fixed_precision(ip, no_decimals=2, factor_multiply=100):
    if isinstance(ip, list):
        return [np.round(i*factor_multiply, no_decimals) for i in ip]
    else:
        return np.round(ip*factor_multiply, no_decimals)


def calc_f1_score_from_conf_mat(conf_mat, label_list, no_results_files, emo2ind_map=None):
    conf_mat = conf_mat/no_results_files
    conf_mat = conf_mat.astype('int')
    epsilon = 0.0000001

    support = np.sum(conf_mat, axis=-1)
    weights = support / (np.sum(support) + epsilon)
    TP = np.diag(conf_mat)
    unweighted_acc = np.sum(TP)/np.sum(support)
    weighted_acc = calc_weighted_acc_from_conf_mat(conf_mat, weights)

    recall = TP / (support + epsilon)
    precision = TP / (np.sum(conf_mat, axis=0) + epsilon)


    classwise_f1 = 2 * np.multiply(precision, recall) / (precision + recall + epsilon)
    micro_f1 =  np.sum(weights*classwise_f1)
    macro_f1 = np.mean(classwise_f1)
    logger.info('conf mat is ')
    logger.info(f'{conf_mat} \n')

    logger.info('conf mat is ')
    logger.info(f'\n {tabulate.tabulate(conf_mat, tablefmt="plain")} \n')
    logger.info(f'{label_list}')
    logger.info(f'\n')

    logger.info(f'support is {support}')

    logger.info(f'classwise_f1: {convert_to_fixed_precision(classwise_f1)}')
    logger.info(f'precision: {convert_to_fixed_precision(precision)}')
    logger.info(f'recall: {convert_to_fixed_precision(recall)}')

    logger.info(f'\n')
    logger.info(f'micro_f1: {convert_to_fixed_precision(micro_f1)}')
    logger.info(f'macro_f1: {convert_to_fixed_precision(macro_f1)}')
    logger.info(f'unweighted_acc: {convert_to_fixed_precision(unweighted_acc)}')
    logger.info(f'weighted_acc: {convert_to_fixed_precision(weighted_acc)}')



# this file is obtained from recover_transcripts_v4.py
Casing_Vs_Punctuation_path = '/export/c04/rpapagari/truecasing_work/data_v3/fisher_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/Casing_Vs_Punctuation_Dataset.csv_test'

Casing_Vs_Punctuation_path = '/export/c04/rpapagari/truecasing_work/data_v3/guetenburg_true_casing_punctuation_WOTurnToken_CombineUCandMC_8classes/Casing_Vs_Punctuation_Dataset.csv_test'


log_file = Casing_Vs_Punctuation_path + '_metrics'
logging.basicConfig(level = logging.INFO,format = '%(message)s', filename=log_file)
logger = logging.getLogger(__name__)



Casing_Vs_Punctuation = pd.read_csv(Casing_Vs_Punctuation_path, sep=',')

label_list = ['AUC', 'LC', 'UC']
num_labels = len(label_list)
Casing_Vs_Punctuation = Casing_Vs_Punctuation[label_list].values.tolist()

conf_mat = np.zeros((num_labels, num_labels))
print(Casing_Vs_Punctuation)

for casing_stat in Casing_Vs_Punctuation:
    pred_class_ind = np.argmax(casing_stat)    
    #import pdb; pdb.set_trace()
    conf_mat[:, pred_class_ind] += casing_stat
    print(conf_mat)


calc_f1_score_from_conf_mat(conf_mat, label_list, 1)


