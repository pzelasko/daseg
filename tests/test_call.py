import pytest

from daseg import Call, FunctionalSegment


@pytest.fixture
def call():
    return Call([
        FunctionalSegment('First segment', 'X', 'me', is_continuation=False),
        FunctionalSegment('Yeah.', 'Y', 'not-me', is_continuation=False),
        FunctionalSegment('has ended', 'X', 'me', is_continuation=True),
    ])


def test_iterate_words(call):
    assert 'First segment Yeah. has ended'.split() == call.words(add_turn_token=False)


def test_iterate_words_with_turns(call):
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == call.words(add_turn_token=True)


def test_iterate_words_with_tags_with_positions(call):
    words, acts = call.words_with_tags(
        add_turn_token=False,
        continuations_allowed=False
    )
    assert 'First segment Yeah. has ended'.split() == words
    assert 'B-X I-X B-Y B-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_continuations(call):
    words, acts = call.words_with_tags(
        add_turn_token=False,
        continuations_allowed=True
    )
    assert 'First segment Yeah. has ended'.split() == words
    assert 'B-X I-X B-Y I-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_turns(call):
    words, acts = call.words_with_tags(
        add_turn_token=True,
        continuations_allowed=False
    )
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == words
    assert 'B-X I-X O B-Y O B-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_turns_with_continuations(call):
    words, acts = call.words_with_tags(
        add_turn_token=True,
        continuations_allowed=True
    )
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == words
    assert 'B-X I-X O B-Y O I-X I-X'.split() == acts


def test_encode_call_separate_coding_no_continuations(call):
    encoded = call.encode(use_joint_coding=False, continuations_allowed=False, add_turn_token=False)
    assert encoded[0].encoded_acts == ['B-X', 'I-X']
    assert encoded[1].encoded_acts == ['B-Y']
    assert encoded[2].encoded_acts == ['B-X', 'I-X']


def test_encode_call_separate_coding_no_continuations_with_turn_token(call):
    encoded = call.encode(use_joint_coding=False, continuations_allowed=False, add_turn_token=True)
    assert encoded[0].encoded_acts == ['B-X', 'I-X']
    assert encoded[1].encoded_acts == ['O']
    assert encoded[2].encoded_acts == ['B-Y']
    assert encoded[3].encoded_acts == ['O']
    assert encoded[4].encoded_acts == ['B-X', 'I-X']


def test_encode_call_separate_coding_with_continuations(call):
    encoded = call.encode(use_joint_coding=False, continuations_allowed=True, add_turn_token=False)
    assert encoded[0].encoded_acts == ['B-X', 'I-X']
    assert encoded[1].encoded_acts == ['B-Y']
    assert encoded[2].encoded_acts == ['I-X', 'I-X']


def test_encode_call_separate_coding_with_continuations_with_turn_token(call):
    encoded = call.encode(use_joint_coding=False, continuations_allowed=True, add_turn_token=True)
    assert encoded[0].encoded_acts == ['B-X', 'I-X']
    assert encoded[1].encoded_acts == ['O']
    assert encoded[2].encoded_acts == ['B-Y']
    assert encoded[3].encoded_acts == ['O']
    assert encoded[4].encoded_acts == ['I-X', 'I-X']


def test_encode_call_joint_coding_no_continuations(call):
    encoded = call.encode(use_joint_coding=True, continuations_allowed=False, add_turn_token=False)
    assert encoded[0].encoded_acts == ['I-', 'X']
    assert encoded[1].encoded_acts == ['Y']
    assert encoded[2].encoded_acts == ['I-', 'X']


def test_encode_call_joint_coding_no_continuations_with_turn_token(call):
    encoded = call.encode(use_joint_coding=True, continuations_allowed=False, add_turn_token=True)
    assert encoded[0].encoded_acts == ['I-', 'X']
    assert encoded[1].encoded_acts == ['O']
    assert encoded[2].encoded_acts == ['Y']
    assert encoded[3].encoded_acts == ['O']
    assert encoded[4].encoded_acts == ['I-', 'X']


def test_encode_call_joint_coding_with_continuations(call):
    encoded = call.encode(use_joint_coding=True, continuations_allowed=True, add_turn_token=False)
    assert encoded[0].encoded_acts == ['I-', 'I-']
    assert encoded[1].encoded_acts == ['Y']
    assert encoded[2].encoded_acts == ['I-', 'X']


def test_encode_call_joint_coding_with_continuations_with_turn_token(call):
    encoded = call.encode(use_joint_coding=True, continuations_allowed=True, add_turn_token=True)
    assert encoded[0].encoded_acts == ['I-', 'I-']
    assert encoded[1].encoded_acts == ['O']
    assert encoded[2].encoded_acts == ['Y']
    assert encoded[3].encoded_acts == ['O']
    assert encoded[4].encoded_acts == ['I-', 'X']
