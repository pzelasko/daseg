from more_itertools import flatten
import sklearn.metrics as sklmetrics
import pickle as pkl
import os, sys
import glob
import numpy as np
from copy import deepcopy
import tabulate
import os, sys
import logging


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
    
    #import pdb; pdb.set_trace()
    #logger.info(metrics_list)
    metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2]
    metrics_list = metrics_list.split(',')
    for metric in metrics_list:
        
        if metric == 'confusion':
            logger.info('per result file conf mat is')
            logger.info(tabulate.tabulate(conf_mat, tablefmt='plain'))
            logger.info(labels_list)
            logger.info('original conf mat is ')
            logger.info(tabulate.tabulate(conf_mat_original, tablefmt='plain'))
            logger.info(labels_list)
        if metric == 'f1-score':
            logger.info(f'classwise_f1: {classwise_f1}')
            logger.info(f'micro_f1: {micro_f1}')
            logger.info(f'macro_f1: {macro_f1}')
        if metric == 'accuracy':
            logger.info(f'unweighted_acc: {unweighted_acc}')                    
    logger.info(classif_report)


def calc_conf_mat_bigdata(y_true, y_pred, label_list, emo2ind_map, debug=False):
    logger.info(f'computing confusion matrix for {len(y_true)} utterances')

    #### inefficient but useful for debugging
    if debug:
        total_conf_mat = np.zeros((len(label_list), len(label_list)))
        for utt_ind in range(len(y_true)):
            conf_mat = sklmetrics.confusion_matrix(y_true[utt_ind], y_pred[utt_ind], labels=label_list)
            total_conf_mat += conf_mat
    
            ######### to check if the metric calculation matches with sklearn tools
            calc_overall_metrics(y_true[utt_ind], y_pred[utt_ind], label_list, 1)
            weighted_acc = calc_weighted_acc_sklearn(y_true[utt_ind], y_pred[utt_ind], emo2ind_map)
            calc_f1_score_from_conf_mat(conf_mat, label_list, 1, emo2ind_map) 
            import pdb; pdb.set_trace()

    ## efficient
    total_conf_mat = sum([sklmetrics.confusion_matrix(y_true[utt_ind], y_pred[utt_ind], labels=label_list)
                                for utt_ind in range(len(y_true))])
    return total_conf_mat


def calc_weighted_acc_sklearn(y_true, y_pred, emo2ind_map):
    y_true_ind = np.array([emo2ind_map[emo] for emo in y_true])
    y_pred_ind = np.array([emo2ind_map[emo] for emo in y_pred])
    no_samples = len(y_true_ind)
    sample_wts = np.ones(no_samples)
    ## obtain weight for each sample in the data
    for idx, i in enumerate(np.bincount(y_true_ind)):
        sample_wts[y_true_ind==idx] *= i/float(no_samples)
    weighted_acc = sklmetrics.accuracy_score(y_true_ind, y_pred_ind, sample_weight=sample_wts)
    #print(f'weighted_acc is {weighted_acc}')
    return weighted_acc


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



results_pkl_path_list = sys.argv[1]
label_block_name = sys.argv[2]
log_file = sys.argv[3]

logging.basicConfig(level = logging.INFO,format = '%(message)s', filename=log_file)
logger = logging.getLogger(__name__)


metrics_list = 'confusion,f1-score,accuracy' #sys.argv[2] 

results_pkl_path_list = results_pkl_path_list.split(',')
results_pkl_path_list = '*'.join(results_pkl_path_list)
results_pkl_path_list = glob.glob(results_pkl_path_list)

logger.info(results_pkl_path_list)
logger.info(f'\n no. of results files are {len(results_pkl_path_list)} \n ')

y_true_all = []
y_pred_all = []
total_conf_mat = []
for results_ind,results_pkl_path in enumerate(results_pkl_path_list):
    logger.info(f'processing {results_pkl_path}')
    results = pkl.load(open(results_pkl_path, 'rb'))    
    target_label_encoder_path = os.path.dirname(results_pkl_path) + '/target_label_encoder.pkl'
    if os.path.exists(target_label_encoder_path):
        target_label_encoder = pkl.load(open(target_label_encoder_path, 'rb'))
        if label_block_name != '-1':
            target_label_encoder = target_label_encoder[label_block_name]

        emo2ind_map = {emo:target_label_encoder.transform([emo])[0] for emo in target_label_encoder.classes_}
        label_list = list(target_label_encoder.classes_)
    else:
        logger.info(f'\n target_label_encoder doesnt exist \n')
        label_list = list(set(flatten(y_true)))
        emo2ind_map = {emo:ind for ind,emo in enumerate(sorted(label_list))}
    ind2emo_map = {ind:emo for emo,ind in emo2ind_map.items()}
    #####################################
    
    #import pdb; pdb.set_trace()

    if label_block_name != '-1':
        true_labels_key = [i for i in results.keys() if i.startswith('true_labels') if i.endswith('_op'+label_block_name)]
        predictions_key = [i for i in results.keys() if i.startswith('predictions') if i.endswith('_op'+label_block_name)]

        true_labels_key = true_labels_key[0]
        predictions_key = predictions_key[0]
    else:
        true_labels_key = 'true_labels'
        predictions_key = 'predictions'
    logger.info(f'{true_labels_key}, {predictions_key}')

    total_conf_mat += [calc_conf_mat_bigdata(results[true_labels_key], results[predictions_key], label_list, emo2ind_map)]


labels_list = list(target_label_encoder.classes_)
logger.info(labels_list)
no_results_files = len(results_pkl_path_list)

total_conf_mat_sum = sum(total_conf_mat)
calc_f1_score_from_conf_mat(total_conf_mat_sum, label_list, no_results_files, emo2ind_map)




