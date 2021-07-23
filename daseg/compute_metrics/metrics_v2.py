from collections import defaultdict
from itertools import chain
from typing import List, Set, Tuple, Dict

import pandas as pd
import seqeval.metrics as seqmetrics
import sklearn
import sklearn.metrics as sklmetrics
import torch
import torch.nn as nn

from Bio import pairwise2
from more_itertools import flatten
import numpy as np
from ordered_set import OrderedSet

from daseg import DialogActCorpus, Call
from daseg.data import CONTINUE_TAG

from pyannote.core import Timeline, Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import DiarizationCoverage

from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.identification import IdentificationPrecision
from pyannote.metrics.identification import IdentificationRecall


def as_tensors(metrics: Dict[str, float]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in metrics.items()}


def compute_sklearn_metrics(true_labels: List[List[str]], predictions: List[List[str]], compute_common_I: bool = True):
    fp = list(flatten(predictions))
    fl = list(flatten(true_labels))
    results = {
        "micro_f1": sklmetrics.f1_score(fl, fp, average='micro'),
        "macro_f1": sklmetrics.f1_score(fl, fp, average='macro'),
    }
    #results.update({"confusion_matrix":sklmetrics.confusion_matrix(fl, fp, labels=list(set(fl)))})
    
    if compute_common_I:
        I_labels = set(l for l in chain(fp, fl) if l.startswith('I-'))
        mapping = {ilab: 'I' for ilab in I_labels}
        fp_common_I = [mapping.get(l, l) for l in fp]
        fl_common_I = [mapping.get(l, l) for l in fl]
        results.update({
            "micro_f1_common_I": sklmetrics.f1_score(fl_common_I, fp_common_I, average='micro'),
            "macro_f1_common_I": sklmetrics.f1_score(fl_common_I, fp_common_I, average='macro'),
        })
    return results


def compute_seqeval_metrics(true_labels: List[List[str]], predictions: List[List[str]]):
    return {
        "precision": seqmetrics.precision_score(true_labels, predictions),
        "recall": seqmetrics.recall_score(true_labels, predictions),
        "f1": seqmetrics.f1_score(true_labels, predictions),
        "accuracy": seqmetrics.accuracy_score(true_labels, predictions),
    }


def compute_zhao_kawahara_metrics_levenshtein(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus):
    """
    Source:
    Zhao, T., & Kawahara, T. (2019). Joint dialog act segmentation and recognition in human conversations using
    attention to dialog context. Computer Speech & Language, 57, 108-127.

    Segmentation metrics:

    DSER (DA Segmentation Error Rate) computes the number of utterances whose boundaries are incorrectly predicted
    divided by the total number of utterances. We regard it as a segment-level segmentation error rate.

    Segmentation WER (Segmentation Word Error Rate) is weighted by the word counts of utterances.
    It computes the number of tokens whose corresponding utterance boundaries are incorrectly predicted divided by the
    total number of tokens.

    Joint metrics:

    DER (DA Error Rate) is similar to the DSER measure, but an utterance is considered to be correct only when its
    boundaries and DA type are all correct.

    Joint WER (Joint Word Error Rate) is similar to the Segmentation WER measure,
    and it also requires both bound- aries and DA type to be correctly predicted.
    """
    span_stats = compute_span_errors(true_dataset=true_dataset, pred_dataset=pred_dataset)
    DSER = (span_stats['sub'] + span_stats['ins'] + span_stats['del']) / span_stats['tot']

    span_stats = compute_span_errors(true_dataset=true_dataset, pred_dataset=pred_dataset, token_weighted=True)
    SegmentationWER = (span_stats['sub'] + span_stats['ins'] + span_stats['del']) / span_stats['tot']

    stats, confusions = compute_labeled_span_errors(true_dataset=true_dataset, pred_dataset=pred_dataset)
    stat_sum = stats.sum(axis=0)
    for col in 'sub ins del'.split():
        if col not in stat_sum:
            stat_sum[col] = 0
    DER = (stat_sum['sub'] + stat_sum['ins'] + stat_sum['del']) / stat_sum['tot']

    stats, _ = compute_labeled_span_errors(true_dataset=true_dataset, pred_dataset=pred_dataset, token_weighted=True)
    stat_sum = stats.sum(axis=0)
    for col in 'sub ins del'.split():
        if col not in stat_sum:
            stat_sum[col] = 0
    JointWER = (stat_sum['sub'] + stat_sum['ins'] + stat_sum['del']) / stat_sum['tot']

    return {
        'DSER': DSER,
        'SegmentationWER': SegmentationWER,
        'DER': DER,
        'JointWER': JointWER,
    }


def compute_labeled_span_errors(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus,
                                token_weighted: bool = False):
    GAP_CHAR = '-'

    alignments = []
    for true_call, pred_call in zip(true_dataset.calls, pred_dataset.calls):
        ali = pairwise2.align.globalxs(
            list(true_call.dialog_act_spans()),
            list(pred_call.dialog_act_spans()),
            -1,
            -1,
            gap_char=[(-1, -1, GAP_CHAR)]
        )[0]
        alignments.append(ali[:2])

    stats = defaultdict(lambda: defaultdict(int))
    confusions = defaultdict(int)
    for ali in alignments:
        for true_span, pred_span in zip(*ali):
            true_da, pred_da = true_span[2], pred_span[2]
            assert not (true_da == GAP_CHAR and pred_da == GAP_CHAR)
            score = true_span[1] - true_span[0] if token_weighted else 1
            if true_da != GAP_CHAR:
                stats['tot'][true_da] += score
            if true_span == pred_span:
                stats['ok'][true_da] += score
            elif true_da == GAP_CHAR:
                stats['ins'][pred_da] += score
            elif pred_da == GAP_CHAR:
                stats['del'][true_da] += score
            else:
                stats['sub'][true_da] += score
                confusions[(true_da, pred_da)] += 1

    stats = pd.DataFrame(stats).fillna(0)
    return stats, confusions


def compute_span_errors(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus, token_weighted: bool = False):
    GAP_CHAR = (-1, -1)

    alignments = []
    for true_call, pred_call in zip(true_dataset.calls, pred_dataset.calls):
        ali = pairwise2.align.globalxs(
            list(true_call.dialog_act_spans(False)),
            list(pred_call.dialog_act_spans(False)),
            -1,
            -1,
            gap_char=[GAP_CHAR]
        )[0]
        alignments.append(ali[:2])

    stats = defaultdict(int)
    for ali in alignments:
        for true_span, pred_span in zip(*ali):
            assert not (true_span == GAP_CHAR and pred_span == GAP_CHAR)
            score = true_span[1] - true_span[0] if token_weighted else 1
            if true_span != GAP_CHAR:
                stats['tot'] += score
            if true_span == pred_span:
                stats['ok'] += score
            elif true_span == GAP_CHAR:
                stats['ins'] += score
            elif pred_span == GAP_CHAR:
                stats['del'] += score
            else:
                stats['sub'] += score
    return stats


def compute_zhao_kawahara_metrics(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus):
    """
    THIS VERSION DOES NOT ALLOW THE ERROR TO EXCEED 100% - IT ONLY COUNTS IF THE REFERENCE SEGMENTS ARE FOUND
    IN THE PREDICTIONS.

    Source:
    Zhao, T., & Kawahara, T. (2019). Joint dialog act segmentation and recognition in human conversations using
    attention to dialog context. Computer Speech & Language, 57, 108-127.

    Segmentation metrics:

    DSER (DA Segmentation Error Rate) computes the number of utterances whose boundaries are incorrectly predicted
    divided by the total number of utterances. We regard it as a segment-level segmentation error rate.

    Segmentation WER (Segmentation Word Error Rate) is weighted by the word counts of utterances.
    It computes the number of tokens whose corresponding utterance boundaries are incorrectly predicted divided by the
    total number of tokens.

    Joint metrics:

    DER (DA Error Rate) is similar to the DSER measure, but an utterance is considered to be correct only when its
    boundaries and DA type are all correct.

    Joint WER (Joint Word Error Rate) is similar to the Segmentation WER measure,
    and it also requires both bound- aries and DA type to be correctly predicted.
    """

    counts = {
        'DSER': 0,
        'SegmentationWER': 0,
        'DER': 0,
        'JointWER': 0,
    }
    total_segments = 0
    total_words = 0

    def build_segment_set(call: Call, with_labels: bool) -> Set[Tuple]:
        segment_set = set()
        word_pos = 0
        for segment in call:
            n_words = len(segment.text.split())
            segment_set.add((word_pos, word_pos + n_words) + ((segment.dialog_act,) if with_labels else ()))
            word_pos += n_words
        return segment_set

    for true_call, pred_call in zip(true_dataset.calls, pred_dataset.calls):
        true_segments = build_segment_set(true_call, with_labels=False)
        pred_segments = build_segment_set(pred_call, with_labels=False)
        error_segments = true_segments - pred_segments
        for segment in error_segments:
            counts['DSER'] += 1
            counts['SegmentationWER'] += segment[1] - segment[0]

        true_segments = build_segment_set(true_call, with_labels=True)
        pred_segments = build_segment_set(pred_call, with_labels=True)
        error_segments = true_segments - pred_segments
        for segment in error_segments:
            counts['DER'] += 1
            counts['JointWER'] += segment[1] - segment[0]

        total_segments += len(true_call)
        total_words += len(true_call.words(add_turn_token=False))

    return {
        'DSER': counts['DSER'] / total_segments,
        'SegmentationWER': counts['SegmentationWER'] / total_words,
        'DER': counts['DER'] / total_segments,
        'JointWER': counts['JointWER'] / total_words
    }


def calculate_pyannote_metrics(true_segments, pred_segments, collar=0.0):
    def convert_plainseg2pyannoteseg(true_segments):
        reference = Annotation()
        for seg in true_segments:
            reference[Segment(seg[0], seg[1])] = seg[2] if len(seg) == 3 else 'unknown'
        return reference
    true_segments_pyannoteseg = convert_plainseg2pyannoteseg(true_segments)
    pred_segments_pyannoteseg = convert_plainseg2pyannoteseg(pred_segments)
    conv_dur_max = 0
    conv_dur_min = 0
    for i in true_segments:
        conv_dur_max = i[1] if conv_dur_max < i[1] else conv_dur_max
        conv_dur_min = i[0] if conv_dur_min > i[0] else conv_dur_min
    arguments = (true_segments_pyannoteseg, pred_segments_pyannoteseg)
    uem = Segment(conv_dur_min, conv_dur_max)

    diarizationErrorRate = DiarizationErrorRate(collar=collar)
    purity = DiarizationPurity(collar=collar)
    coverage = DiarizationCoverage(collar=collar)
    identificationErrorRate = IdentificationErrorRate(collar=collar)
    precision = IdentificationPrecision(collar=collar)
    recall = IdentificationRecall(collar=collar)

    model_optimal_map = diarizationErrorRate.optimal_mapping(*arguments)
    model_purity = purity(*arguments, uem=uem)
    model_coverage = coverage(*arguments, uem=uem)

    model_ier  = identificationErrorRate(*arguments, uem=uem, detailed=True)
    model_precision = precision(*arguments, uem=uem)
    model_recall = recall(*arguments, uem=uem)

    return {
        'model_purity': model_purity,
        'model_coverage': model_coverage,
        'model_optimal_map': model_optimal_map,
        'model_ier': model_ier,
        'model_precision': model_precision,
        'model_recall': model_recall
    }


def compute_zhao_kawahara_metrics_speech(true_labels, pred_labels, true_seg_boundaries, label_scheme, collar=0.0, ignore_labels=['OOS', 'silence']):
    """
    THIS VERSION DOES NOT ALLOW THE ERROR TO EXCEED 100% - IT ONLY COUNTS IF THE REFERENCE SEGMENTS ARE FOUND
    IN THE PREDICTIONS.

    Source:
    Zhao, T., & Kawahara, T. (2019). Joint dialog act segmentation and recognition in human conversations using
    attention to dialog context. Computer Speech & Language, 57, 108-127.

    Segmentation metrics:

    DSER (DA Segmentation Error Rate) computes the number of utterances whose boundaries are incorrectly predicted
    divided by the total number of utterances. We regard it as a segment-level segmentation error rate.

    Segmentation WER (Segmentation Word Error Rate) is weighted by the word counts of utterances.
    It computes the number of tokens whose corresponding utterance boundaries are incorrectly predicted divided by the
    total number of tokens.

    Joint metrics:

    DER (DA Error Rate) is similar to the DSER measure, but an utterance is considered to be correct only when its
    boundaries and DA type are all correct.

    Joint WER (Joint Word Error Rate) is similar to the Segmentation WER measure,
    and it also requires both bound- aries and DA type to be correctly predicted.
    """

    counts = {
        'DSER': 0,
        'SegmentationWER': 0,
        'DER': 0,
        'JointWER': 0,
    }
    total_segments = 0
    total_frames = 0

    def build_segment_set(targets, seg_boundaries, with_labels: bool):
        segment_set = OrderedSet()
        #segment_set = set()
        bounday_ind = [ind for ind,segment_flag in enumerate(seg_boundaries) if segment_flag]
        start = 0
        for end in bounday_ind:
            label = targets[end] #label_map[targets[end]]
            segment_set.add((start, end) + ((label,) if with_labels else ()))
            start = end #+ 1
        return segment_set

    def targets2segmentset(targets, label_scheme, with_labels: bool):
        '''First get segment boundaries from targets then build segment set
        '''
        if label_scheme == 'Exact':
            new_targets = targets
        elif label_scheme == 'E':
            intermediate_label = 'I-'
            prev_label = targets[-1]
            new_targets = []
            for pred in targets[::-1]:
                if pred != intermediate_label:
                    prev_label = pred
                new_targets.append(prev_label)
            new_targets = new_targets[::-1]
        elif label_scheme == 'IE':
            new_targets = [pred[2:] if pred.startswith('I-') else pred for pred in targets if pred.startswith('I-')]

        seg_boundaries = [i!=j for i,j in zip(new_targets[1:], new_targets[:-1])]
        seg_boundaries += [True]   # last frame/token has to be a segment boundary 
        segments_count = sum(seg_boundaries)
        return build_segment_set(targets, seg_boundaries, with_labels), segments_count

    
    true_labels_all = list(flatten(true_labels))
    pred_labels_all = list(flatten(pred_labels))

    if ignore_labels:
        valid_ind = [ind for ind,label in enumerate(true_labels_all) if not label in ignore_labels]

    for with_labels in [True]:
        
        true_segments, true_segments_count = targets2segmentset(true_labels_all, label_scheme, with_labels=with_labels)
        pred_segments, pred_segments_count = targets2segmentset(pred_labels_all, label_scheme, with_labels=with_labels)
        pyannote_results = calculate_pyannote_metrics(true_segments, pred_segments, collar=collar)
        for i,j in pyannote_results.items():
            print(i, j)
       
        if ignore_labels:
            true_labels_all = [true_labels_all[ind] for ind in valid_ind]
            pred_labels_all = [pred_labels_all[ind] for ind in valid_ind]
            true_segments, true_segments_count = targets2segmentset(true_labels_all, label_scheme, with_labels=with_labels)
            pred_segments, pred_segments_count = targets2segmentset(pred_labels_all, label_scheme, with_labels=with_labels)       
 
            pyannote_results_OnlyMainEmo = calculate_pyannote_metrics(true_segments, pred_segments, collar=collar)
            pyannote_results_new = {i+'_OnlyMainEmo':j  for i,j in pyannote_results_OnlyMainEmo.items()}
            for i,j in pyannote_results_new.items():
                print(i, j)
            pyannote_results.update(pyannote_results_new)
    return pyannote_results


    #for true_utt_labels, pred_utt_labels in zip(true_labels, pred_labels):
    #    true_segments, true_segments_count = targets2segmentset(true_utt_labels, label_scheme, with_labels=False)
    #    pred_segments, pred_segments_count = targets2segmentset(pred_utt_labels, label_scheme, with_labels=False)
    #    
    #    #import pdb; pdb.set_trace()
    #    #model_purity, model_coverage, model_optimal_map, model_ier, model_precision, model_recall = calculate_pyannote_metrics(true_segments, pred_segments)
    #    #decisions = [true_utt_labels[i]==pred_utt_labels[i] for i in range(len(true_utt_labels))]
    #    #print(np.sum(decisions))

    #    error_segments = true_segments - pred_segments
    #    for segment in error_segments:
    #        counts['DSER'] += 1
    #        counts['SegmentationWER'] += segment[1] - segment[0]

    #    true_segments, true_segments_count = targets2segmentset(true_utt_labels, label_scheme, with_labels=True)
    #    pred_segments, pred_segments_count = targets2segmentset(pred_utt_labels, label_scheme, with_labels=True)
    #    #model_purity, model_coverage, model_optimal_map, model_ier, model_precision, model_recall = calculate_pyannote_metrics(true_segments, pred_segments)

    #    error_segments = true_segments - pred_segments
    #    for segment in error_segments:
    #        counts['DER'] += 1
    #        counts['JointWER'] += segment[1] - segment[0]

    #    total_segments += true_segments_count
    #    total_frames += len(true_utt_labels)

    #return {
    #    'DSER': counts['DSER'] / total_segments,
    #    'SegmentationWER': counts['SegmentationWER'] / total_frames,
    #    'DER': counts['DER'] / total_segments,
    #    'JointWER': counts['JointWER'] / total_frames,
    #    'model_purity': model_purity,
    #    'model_coverage': model_coverage,
    #    'model_optimal_map': model_optimal_map,
    #    'model_ier': model_ier,
    #    'model_precision': model_precision,
    #    'model_recall': model_recall
    #}


############ obtaining conversational level labels, both ground truth and model predictions  #################
def conv_level_emo_pred(logits, y_true, emo2ind_map, ind2emo_map, ignore_neu=False):
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

    #print(list(zip(conv_true_pred, conv_model_pred)))
    conv_level_emo_acc = np.mean([i==j for i,j in zip(conv_true_pred, conv_model_pred)])
    print(f'conv_level_emo_acc is {conv_level_emo_acc}')
    return conv_true_pred, conv_model_pred, conv_level_emo_acc


def conv_level_senti_pred(conv_true_pred, conv_model_pred, emo2sentiment_map):
    ### process emotions as they are and at the end convert them to sentiments for performance eval
    conv_true_senti_pred = [emo2sentiment_map[i] for i in conv_true_pred]
    conv_model_senti_pred = [emo2sentiment_map[i] for i in conv_model_pred]
    conv_level_senti_acc = np.mean([i==j for i,j in zip(conv_true_senti_pred, conv_model_senti_pred)])
    print(f'conv level senti acc is {conv_level_senti_acc}')
    return conv_true_senti_pred, conv_model_senti_pred, conv_level_senti_acc


#def conv_level_senti_pred(logits, y_true, emo2ind_map, ind2emo_map, emo2sentiment_map):
#    ### process emotions as they are and at the end convert them to sentiments for performance eval
#    logits = torch.tensor(logits)
#    model_posteriors = nn.functional.softmax(logits, dim=-1)
#    model_posteriors = model_posteriors.to('cpu').numpy()
#    conv_model_pred = np.sum(model_posteriors, axis=1) # sum along time dimension
#    conv_model_pred = np.argmax(conv_model_pred, axis=-1)
#    conv_model_pred_emo = [ind2emo_map[i] for i in conv_model_pred]
#    conv_model_pred_sentiment = [emo2sentiment_map[i] for i in conv_model_pred_emo]
#
#    conv_true_pred = []
#    for utt_ind in range(len(y_true)):
#        utt_labels = y_true[utt_ind]
#        most_freq_emo = max(set(utt_labels), key=utt_labels.count)
#        conv_true_pred.append(most_freq_emo)   
#    conv_true_pred_sentiment = [emo2sentiment_map[i] for i in conv_true_pred] 
#
#    conv_level_senti_acc =  np.mean([i==j for i,j in zip(conv_true_pred_sentiment, conv_model_pred_sentiment)])
#    return conv_true_pred_sentiment, conv_model_pred_sentiment, 


def compute_zhao_kawahara_metrics_DASEG_analysis(true_labels, pred_labels, true_seg_boundaries, label_scheme, collar=0.0):
    counts = {
        'DSER': 0,
        'SegmentationWER': 0,
        'DER': 0,
        'JointWER': 0,
    }
    total_segments = 0
    total_frames = 0

    def build_segment_set(targets, seg_boundaries, with_labels: bool):
        segment_set = OrderedSet()
        #segment_set = set()
        bounday_ind = [ind for ind,segment_flag in enumerate(seg_boundaries) if segment_flag]
        start = 0
        for end in bounday_ind:
            label = targets[end] #label_map[targets[end]]
            segment_set.add((start, end) + ((label,) if with_labels else ()))
            start = end #+ 1
        return segment_set

    def targets2segmentset(targets, label_scheme, with_labels: bool):
        '''First get segment boundaries from targets then build segment set
        '''
        ignore_indices = []
        if label_scheme == 'Exact':
            new_targets = targets
        elif label_scheme == 'E':
            intermediate_label = 'I-'
            prev_label = targets[-1]
            new_targets = []
            #total_target_length = len(targets)
            
            #targets = [i for i in targets if i!='O']
            for ind,pred in enumerate(targets[::-1]):
                if pred != intermediate_label:
                    prev_label = pred
                new_targets.append(prev_label)    
            new_targets = new_targets[::-1]
        elif label_scheme == 'IE':
            new_targets = [pred[2:] if pred.startswith('I-') else pred for pred in targets if pred.startswith('I-')]
        import pdb; pdb.set_trace()
        seg_boundaries = [i!=j for i,j in zip(new_targets[1:], new_targets[:-1])]
        seg_boundaries += [True]   # last frame/token has to be a segment boundary 
        segments_count = sum(seg_boundaries)
        return build_segment_set(targets, seg_boundaries, with_labels), segments_count

    
    #true_labels_all = list(flatten(true_labels))
    #pred_labels_all = list(flatten(pred_labels))
    ##true_seg_boundaries_all = list(np.concatenate(true_seg_boundaries, axis=-1))
    #for with_labels in [True]:
    #    
    #    true_segments, true_segments_count = targets2segmentset(true_labels_all, label_scheme, with_labels=with_labels)
    #    pred_segments, pred_segments_count = targets2segmentset(pred_labels_all, label_scheme, with_labels=with_labels)
 
    #    model_purity, model_coverage, model_optimal_map, model_ier, model_precision, model_recall = calculate_pyannote_metrics(true_segments, pred_segments, collar=collar)
    #    print(f'Purity: {model_purity}, Coverage: {model_coverage}')
    #    print(f'Optimap_map: {model_optimal_map}')
    #    print(model_ier)
    #    print(f'Precision: {model_precision}, Recall: {model_recall}')

    for true_utt_labels, pred_utt_labels in zip(true_labels, pred_labels):
        #true_segments, true_segments_count = targets2segmentset(true_utt_labels, label_scheme, with_labels=False)
        #pred_segments, pred_segments_count = targets2segmentset(pred_utt_labels, label_scheme, with_labels=False)
        
        #import pdb; pdb.set_trace()
        #model_purity, model_coverage, model_optimal_map, model_ier, model_precision, model_recall = calculate_pyannote_metrics(true_segments, pred_segments)
        #decisions = [true_utt_labels[i]==pred_utt_labels[i] for i in range(len(true_utt_labels))]
        #print(np.sum(decisions))

        #error_segments = true_segments - pred_segments
        #for segment in error_segments:
        #    counts['DSER'] += 1
        #    counts['SegmentationWER'] += segment[1] - segment[0]

        import pdb; pdb.set_trace()
        keep_indices = [ind for ind,i in enumerate(true_utt_labels) if i!='O']
        true_utt_labels = [true_utt_labels[i] for i in keep_indices]
        pred_utt_labels = [pred_utt_labels[i] for i in keep_indices]
        true_segments, true_segments_count = targets2segmentset(true_utt_labels, label_scheme, with_labels=True)
        import pdb; pdb.set_trace()
        pred_segments, pred_segments_count = targets2segmentset(pred_utt_labels, label_scheme, with_labels=True)
        #model_purity, model_coverage, model_optimal_map, model_ier, model_precision, model_recall = calculate_pyannote_metrics(true_segments, pred_segments)

        error_segments = true_segments - pred_segments
        for segment in error_segments:
            counts['DER'] += 1
            counts['JointWER'] += segment[1] - segment[0]

        total_segments += true_segments_count
        total_frames += len(true_utt_labels)

    return {
        'DSER': counts['DSER'] / total_segments,
        'SegmentationWER': counts['SegmentationWER'] / total_frames,
        'DER': counts['DER'] / total_segments,
        'JointWER': counts['JointWER'] / total_frames,
        #'model_purity': model_purity,
        #'model_coverage': model_coverage,
        #'model_optimal_map': model_optimal_map,
        #'model_ier': model_ier,
        #'model_precision': model_precision,
        #'model_recall': model_recall
    }


"""
The original Zhao-Kawahara metrics, code adapted to this codebase, is below.
"""


def is_end_tag(da_tag):
    return da_tag != CONTINUE_TAG


def instsance_segmentation_metrics(true_labels, pred_labels):
    assert len(true_labels) == len(pred_labels)
    n_ref_units = 0.0
    n_words = float(len(true_labels))
    n_words_in_wrong_unit = 0.0
    n_wrong_units = 0.0

    last_ref_b_index = 0
    last_pred_b_index = 0
    this_unit_is_wrong = False
    for index, ref in enumerate(true_labels):
        # ref = ref[0]
        # pred = pred_labels[index][0]
        pred = pred_labels[index]

        if is_end_tag(ref):  # ref boundaries
            n_ref_units += 1

            if not is_end_tag(pred) or last_ref_b_index != last_pred_b_index:
                this_unit_is_wrong = True
            if this_unit_is_wrong:
                n_wrong_units += 1
                n_words_in_wrong_unit += (index - last_ref_b_index)
                this_unit_is_wrong = False
            last_ref_b_index = index
        if is_end_tag(pred):
            last_pred_b_index = index

    if n_ref_units == 0.0:
        dser = 1.0
    else:
        dser = n_wrong_units / n_ref_units
    strict_err = n_words_in_wrong_unit / n_words

    return dser, strict_err


def instsance_da_metrics(true_labels, pred_labels):
    n_ref_units = 0.0
    n_words = float(len(true_labels))
    n_words_in_wrong_unit_or_da = 0.0
    n_wrong_units = 0.0

    last_ref_b_index = 0
    last_pred_b_index = 0
    this_unit_is_wrong = False
    for index, ref in enumerate(true_labels):
        pred = pred_labels[index]
        # ref_seg = ref[0]
        # ref_da = ref[2:]
        # pred_seg = pred[0]
        # pred_da = pred[2:]
        ref_seg = ref
        ref_da = ref
        pred_seg = pred
        pred_da = pred

        if is_end_tag(ref_seg):  # ref boundaries
            n_ref_units += 1

            if not is_end_tag(pred_seg) or last_ref_b_index != last_pred_b_index:
                this_unit_is_wrong = True
            elif ref_da != pred_da:
                this_unit_is_wrong = True
            if this_unit_is_wrong:
                n_wrong_units += 1
                n_words_in_wrong_unit_or_da += (index - last_ref_b_index)
                this_unit_is_wrong = False
            last_ref_b_index = index
        if is_end_tag(pred_seg):
            last_pred_b_index = index

    if n_ref_units == 0.0:
        der = 1.0
    else:
        der = n_wrong_units / n_ref_units
    strict_err = n_words_in_wrong_unit_or_da / n_words

    return der, strict_err


def segmentation_metrics(true_labels_lst, pred_labels_lst):
    dser = 0.0
    strict_err = 0.0

    for i in range(len(true_labels_lst)):
        tmp_seg_metrics = instsance_segmentation_metrics(true_labels_lst[i], pred_labels_lst[i])
        dser += tmp_seg_metrics[0]
        strict_err += tmp_seg_metrics[1]

    dser /= len(true_labels_lst)
    strict_err /= len(true_labels_lst)

    flatten_true_labels = []
    flatten_pred_labels = []
    for l in true_labels_lst:
        seg_l = [label[0] for label in l]
        flatten_true_labels += seg_l
    for l in pred_labels_lst:
        seg_l = [label[0] for label in l]
        flatten_pred_labels += seg_l
    macro_f1 = sklearn.metrics.f1_score(flatten_true_labels, flatten_pred_labels, average="macro")
    micro_f1 = sklearn.metrics.f1_score(flatten_true_labels, flatten_pred_labels, average="micro")

    return {'DSER': dser, 'SegmentationWER': strict_err, 'macro_f1': macro_f1, 'micro_f1': micro_f1}


def da_metrics(true_labels_lst, pred_labels_lst):
    der = 0.0
    strict_err = 0.0

    for i in range(len(true_labels_lst)):
        tmp_da_metrics = instsance_da_metrics(true_labels_lst[i], pred_labels_lst[i])
        der += tmp_da_metrics[0]
        strict_err += tmp_da_metrics[1]
    der /= len(true_labels_lst)
    strict_err /= len(true_labels_lst)

    flatten_true_labels = []
    flatten_pred_labels = []
    for l in true_labels_lst:
        flatten_true_labels += l
    for l in pred_labels_lst:
        flatten_pred_labels += l
    macro_f1 = sklearn.metrics.f1_score(flatten_true_labels, flatten_pred_labels, average="macro")
    micro_f1 = sklearn.metrics.f1_score(flatten_true_labels, flatten_pred_labels, average="micro")

    return {'DER': der, 'JointWER': strict_err, 'macro_f1': macro_f1, 'micro_f1': micro_f1}


def compute_original_zhao_kawahara_metrics(true_turns: List[List[str]], pred_turns: List[List[str]]) -> dict:
    all_metrics = da_metrics(true_labels_lst=true_turns, pred_labels_lst=pred_turns)
    seg_metrics = segmentation_metrics(true_labels_lst=true_turns, pred_labels_lst=pred_turns)
    all_metrics['DSER'] = seg_metrics['DSER']
    all_metrics['SegmentationWER'] = seg_metrics['SegmentationWER']
    return all_metrics
