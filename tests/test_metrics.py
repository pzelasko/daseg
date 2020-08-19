from functools import reduce
from operator import add
from typing import List

import pytest

from daseg import DialogActCorpus, Call, FunctionalSegment
from daseg.metrics import compute_zhao_kawahara_metrics, compute_original_zhao_kawahara_metrics


def as_labels(corpus: DialogActCorpus) -> List[List[str]]:
    return [
        reduce(
            add,
            (s.encoded_acts for s in Call(turn).encode(
                use_joint_coding=True, continuations_allowed=False, add_turn_token=False
            ))
        )
        for turn in corpus.turns
    ]


@pytest.fixture
def true_dataset():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a b c', dialog_act='sd', speaker='A'),
            FunctionalSegment('a b c', dialog_act='ad', speaker='A'),
            FunctionalSegment('a b c', dialog_act='h', speaker='A'),
            FunctionalSegment('a b', dialog_act='qy', speaker='A'),
        ])
    })


@pytest.fixture
def pred_dataset():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a b c', dialog_act='sd', speaker='A'),
            FunctionalSegment('a b c a', dialog_act='ad', speaker='A'),
            FunctionalSegment('b c', dialog_act='h', speaker='A'),
            FunctionalSegment('a b', dialog_act='qy^d', speaker='A'),
        ])
    })


def test_zhao_kwahara_metrics(true_dataset, pred_dataset):
    metrics = compute_zhao_kawahara_metrics(true_dataset=true_dataset, pred_dataset=pred_dataset)
    assert metrics['DSER'] == 2 / 4
    assert metrics['SegmentationWER'] == 6 / 11
    assert metrics['DER'] == 3 / 4
    assert metrics['JointWER'] == 8 / 11


def test_original_zhao_kwahara_metrics(true_dataset, pred_dataset):
    metrics = compute_original_zhao_kawahara_metrics(
        true_turns=as_labels(true_dataset),
        pred_turns=as_labels(pred_dataset)
    )
    assert metrics['DSER'] == 2 / 4
    assert metrics['SegmentationWER'] == 6 / 11
    assert metrics['DER'] == 3 / 4
    assert metrics['JointWER'] == 8 / 11


@pytest.fixture
def true_dataset_ins():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a b c', dialog_act='sd', speaker='A'),
        ])
    })


@pytest.fixture
def pred_dataset_ins():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a', dialog_act='sd', speaker='A'),
            FunctionalSegment('b c', dialog_act='sd', speaker='A'),
        ])
    })


@pytest.fixture
def pred_dataset_ins_diff_label():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a', dialog_act='sv', speaker='A'),
            FunctionalSegment('b c', dialog_act='sd', speaker='A'),
        ])
    })


def test_zhao_kwahara_metrics_segment_insertion(true_dataset_ins, pred_dataset_ins):
    metrics = compute_zhao_kawahara_metrics(true_dataset=true_dataset_ins, pred_dataset=pred_dataset_ins)
    assert metrics['DSER'] == 1 / 1
    assert metrics['SegmentationWER'] == 3 / 3
    assert metrics['DER'] == 1 / 1
    assert metrics['JointWER'] == 3 / 3


def test_zhao_kwahara_metrics_segment_insertion_different_label(true_dataset_ins, pred_dataset_ins_diff_label):
    metrics = compute_zhao_kawahara_metrics(true_dataset=true_dataset_ins, pred_dataset=pred_dataset_ins_diff_label)
    assert metrics['DSER'] == 1 / 1
    assert metrics['SegmentationWER'] == 3 / 3
    assert metrics['DER'] == 1 / 1
    assert metrics['JointWER'] == 3 / 3


def test_original_zhao_kwahara_metrics_segment_insertion(true_dataset_ins, pred_dataset_ins):
    metrics = compute_original_zhao_kawahara_metrics(
        true_turns=as_labels(true_dataset_ins),
        pred_turns=as_labels(pred_dataset_ins)
    )
    assert metrics['DSER'] == 1 / 1
    assert metrics['SegmentationWER'] == 3 / 3
    assert metrics['DER'] == 1 / 1
    assert metrics['JointWER'] == 3 / 3


def test_original_zhao_kwahara_metrics_segment_insertion_different_label(true_dataset_ins, pred_dataset_ins_diff_label):
    metrics = compute_original_zhao_kawahara_metrics(
        true_turns=as_labels(true_dataset_ins),
        pred_turns=as_labels(pred_dataset_ins_diff_label)
    )
    assert metrics['DSER'] == 1 / 1
    assert metrics['SegmentationWER'] == 3 / 3
    assert metrics['DER'] == 1 / 1
    assert metrics['JointWER'] == 3 / 3
