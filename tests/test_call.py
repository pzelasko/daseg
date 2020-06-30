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


@pytest.mark.parametrize('continuations_allowed', [True, False])
def test_iterate_words_with_tags(call, continuations_allowed):
    words, acts = call.words_with_tags(
        add_turn_token=False,
        indicate_begin_continue=False,
        continuations_allowed=continuations_allowed
    )
    assert 'First segment Yeah. has ended'.split() == words
    assert 'X X Y X X'.split() == acts


@pytest.mark.parametrize('continuations_allowed', [True, False])
def test_iterate_words_with_tags_with_turns(call, continuations_allowed):
    words, acts = call.words_with_tags(
        add_turn_token=True,
        indicate_begin_continue=False,
        continuations_allowed=continuations_allowed
    )
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == words
    assert 'X X O Y O X X'.split() == acts


def test_iterate_words_with_tags_with_positions(call):
    words, acts = call.words_with_tags(
        add_turn_token=False,
        indicate_begin_continue=True,
        continuations_allowed=False
    )
    assert 'First segment Yeah. has ended'.split() == words
    assert 'B-X I-X B-Y B-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_continuations(call):
    words, acts = call.words_with_tags(
        add_turn_token=False,
        indicate_begin_continue=True,
        continuations_allowed=True
    )
    assert 'First segment Yeah. has ended'.split() == words
    assert 'B-X I-X B-Y I-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_turns(call):
    words, acts = call.words_with_tags(
        add_turn_token=True,
        indicate_begin_continue=True,
        continuations_allowed=False
    )
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == words
    assert 'B-X I-X O B-Y O B-X I-X'.split() == acts


def test_iterate_words_with_tags_with_positions_with_turns_with_continuations(call):
    words, acts = call.words_with_tags(
        add_turn_token=True,
        indicate_begin_continue=True,
        continuations_allowed=True
    )
    assert 'First segment <TURN> Yeah. <TURN> has ended'.split() == words
    assert 'B-X I-X O B-Y O I-X I-X'.split() == acts
