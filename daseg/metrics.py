from collections import defaultdict
from itertools import chain
from typing import Dict, List, Set, Tuple

import pandas as pd
import seqeval.metrics as seqmetrics
import sklearn
import sklearn.metrics as sklmetrics
import torch
from Bio import pairwise2
from more_itertools import flatten

from daseg import Call, DialogActCorpus
from daseg.data import CONTINUE_TAG


def as_tensors(metrics: Dict[str, float]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in metrics.items()}


def validate_ids(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus):
    true_ids = set(true_dataset.call_ids)
    pred_ids = set(pred_dataset.call_ids)
    not_in_true = pred_ids - true_ids
    not_in_pred = true_ids - pred_ids
    assert len(not_in_true) == 0, f"There are {not_in_true} calls with IDs missing in the true set ({not_in_true})."
    assert len(not_in_pred) == 0, f"There are {not_in_pred} calls with IDs missing in the pred set ({not_in_pred})."


def compute_sklearn_metrics(true_labels: List[List[str]], predictions: List[List[str]], compute_common_I: bool = True):
    fp = list(flatten(predictions))
    fl = list(flatten(true_labels))
    results = {
        "micro_f1": sklmetrics.f1_score(fl, fp, average='micro'),
        "macro_f1": sklmetrics.f1_score(fl, fp, average='macro'),
    }
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


def compute_segeval_metrics(true_dataset: DialogActCorpus, pred_dataset: DialogActCorpus):
    from statistics import mean
    from segeval import boundary_similarity, pk
    from segeval.data import Dataset
    from segeval.compute import summarize

    def fix_single_seg_calls(true, pred):
        for cid in true.keys():
            true_segs = true[cid]["ref"]
            pred_segs = pred[cid]["hyp"]
            if len(true_segs) == len(pred_segs) == 1:
                true[cid]["ref"] = true_segs + [1]
                pred[cid]["hyp"] = pred_segs + [1]

    true_segments = {
        cid: {"ref": [len(fs.text.split()) for fs in call]}
        for cid, call in true_dataset.dialogues.items()
    }
    pred_segments = {
        cid: {"hyp": [len(fs.text.split()) for fs in call]}
        for cid, call in pred_dataset.dialogues.items()
    }

    fix_single_seg_calls(true_segments, pred_segments)

    pred_segments = Dataset(pred_segments)
    true_segments = Dataset(true_segments)

    return {
        "pk": float(mean(pk(true_segments, pred_segments).values())),
        "B(tol=2)": summarize(boundary_similarity(true_segments, pred_segments)),
        "B(tol=5)": summarize(boundary_similarity(true_segments, pred_segments, n_t=5)),
        "B(tol=10)": summarize(boundary_similarity(true_segments, pred_segments, n_t=10)),
        # "CM": summarize(boundary_confusion_matrix(true_segments, pred_segments)),
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
    validate_ids(true_dataset=true_dataset, pred_dataset=pred_dataset)
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
    for cid in true_dataset.call_ids:
        true_call = true_dataset[cid]
        pred_call = pred_dataset[cid]
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
    for cid in true_dataset.call_ids:
        true_call = true_dataset[cid]
        pred_call = pred_dataset[cid]
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
    validate_ids(true_dataset=true_dataset, pred_dataset=pred_dataset)

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

    for cid in true_dataset.call_ids:
        true_call = true_dataset[cid]
        pred_call = pred_dataset[cid]
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
