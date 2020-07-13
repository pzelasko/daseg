from collections import defaultdict
from itertools import chain
from typing import List

import pandas as pd
import seqeval.metrics as seqmetrics
import sklearn.metrics as sklmetrics
from Bio import pairwise2
from more_itertools import flatten

from daseg import DialogActCorpus


def compute_sklearn_metrics(true_labels: List[List[str]], predictions: List[List[str]]):
    fp = list(flatten(predictions))
    fl = list(flatten(true_labels))

    I_labels = set(l for l in chain(fp, fl) if l.startswith('I-'))
    mapping = {ilab: 'I' for ilab in I_labels}

    fp_common_I = [mapping.get(l, l) for l in fp]
    fl_common_I = [mapping.get(l, l) for l in fl]
    return {
        "micro_precision": sklmetrics.precision_score(fl, fp, average='micro'),
        "micro_recall": sklmetrics.recall_score(fl, fp, average='micro'),
        "micro_f1": sklmetrics.f1_score(fl, fp, average='micro'),
        "macro_precision": sklmetrics.precision_score(fl, fp, average='macro'),
        "macro_recall": sklmetrics.recall_score(fl, fp, average='macro'),
        "macro_f1": sklmetrics.f1_score(fl, fp, average='macro'),
        "accuracy": sklmetrics.accuracy_score(fl, fp),
        "micro_f1_common_I": sklmetrics.f1_score(fl_common_I, fp_common_I, average='micro'),
        "macro_f1_common_I": sklmetrics.f1_score(fl_common_I, fp_common_I, average='macro'),
    }


def compute_seqeval_metrics(true_labels: List[List[str]], predictions: List[List[str]]):
    return {
        "precision": seqmetrics.precision_score(true_labels, predictions),
        "recall": seqmetrics.recall_score(true_labels, predictions),
        "f1": seqmetrics.f1_score(true_labels, predictions),
        "accuracy": seqmetrics.accuracy_score(true_labels, predictions),
    }


def compute_zhao_kawahara_metrics(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus):
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
            if true_da == pred_da:
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
