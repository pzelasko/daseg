from functools import lru_cache

import pytest

from daseg import SwdaDataset, Call, FunctionalSegment


@lru_cache(1)
def dummy_dataset():
    return SwdaDataset(dialogues={
        'call0': Call([
            FunctionalSegment('Hi, how are you?', 'Conventional-Opening', 'A'),
            FunctionalSegment("I'm fine, thanks. And you?", 'Conventional-Opening', 'B'),
            FunctionalSegment("Good.", 'Conventional-Opening', 'A'),
            FunctionalSegment("It's just a test.", 'Statement-non-opinion', 'A'),
            FunctionalSegment("How do you know?", 'Wh-Question', 'B'),
        ])
    })


def test_attributes():
    dataset = dummy_dataset()
    assert set(dataset.dialog_acts) == {'Conventional-Opening', 'Statement-non-opinion', 'Wh-Question'}
    assert set(dataset.dialog_act_labels) == {
        'B-Conventional-Opening', 'I-Conventional-Opening',
        'B-Statement-non-opinion', 'I-Statement-non-opinion',
        'B-Wh-Question', 'I-Wh-Question',
        'O'
    }
    assert dataset.call_ids == ['call0']
    assert len(dataset.calls) == 1


@pytest.mark.parametrize(
    ['call_id', 'expected_num_calls'],
    [
        ('call0', 1),
        ('out-of-set', 0)
    ])
def test_subset(call_id, expected_num_calls):
    dataset = dummy_dataset()
    assert len(dataset.subset([call_id]).calls) == expected_num_calls
