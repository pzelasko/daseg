import pytest
from transformers import AutoTokenizer

from daseg.data import prepare_call_windows, Call, FunctionalSegment
from daseg.dataloaders.transformers import as_windows


@pytest.mark.parametrize(
    ['seq', 'length', 'overlap', 'expected'],
    [
        ([0, 1, 2, 3, 4, 5], None, None, [[0, 1, 2, 3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 3, 0, [[0, 1, 2], [3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 3, 1, [[0, 1, 2], [2, 3, 4], [3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 3, 2, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 4, 0, [[0, 1, 2, 3], [2, 3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 4, 1, [[0, 1, 2, 3], [2, 3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 4, 2, [[0, 1, 2, 3], [2, 3, 4, 5]]),
        ([0, 1, 2, 3, 4, 5], 4, 3, [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
    ]
)
def test_call_windows(seq, length, overlap, expected):
    windows = prepare_call_windows(call=seq, acts_count_per_sample=length, acts_count_overlap=overlap)
    assert expected == windows


@pytest.fixture
def a_call():
    return Call([
        FunctionalSegment('Hi, how are you?', 'Conventional-Opening', 'A'),
        FunctionalSegment("I'm fine, thanks. And you?", 'Conventional-Opening', 'B'),
        FunctionalSegment("Good.", 'Conventional-Opening', 'A'),
        FunctionalSegment("It's just a test.", 'Statement-non-opinion', 'A'),
        FunctionalSegment("How do you know?", 'Wh-Question', 'B'),
    ])


def test_as_windows(a_call):
    windows = list(as_windows(
        call=a_call,
        max_length=20,
        tokenizer=AutoTokenizer.from_pretrained('roberta-base'),
        use_joint_coding=True
    ))
    assert len(windows) == 2
    assert a_call[:3] == windows[0]
    assert a_call[3:] == windows[1]
