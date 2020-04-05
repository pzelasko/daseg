import pytest

from daseg.data import prepare_call_windows


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
