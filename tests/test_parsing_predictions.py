import pytest

from daseg import DialogActCorpus, Call, FunctionalSegment
from daseg.conversion import predictions_to_dataset, joint_coding_predictions_to_turns, \
    joint_coding_predictions_to_corpus


@pytest.fixture
def dataset():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('hi how are you', dialog_act='Question', speaker='A'),
            FunctionalSegment('long time no see', dialog_act='Statement', speaker='A'),
            FunctionalSegment("i'm fine", dialog_act='Statement', speaker='B'),
            FunctionalSegment("and you", dialog_act='Question', speaker='B'),
        ])
    })


@pytest.fixture
def predictions():
    return [
        'B-Question I-Question I-Question I-Question'.split() +
        'B-Statement I-Statement I-Statement I-Statement O'.split() +
        'B-Backchannel I-Statement'.split() +
        'B-Question I-Question'.split()
    ]


@pytest.fixture
def joint_coding_predictions():
    return [
        'I- I- I- Question'.split() +
        'I- I- I- Statement O'.split() +
        'Backchannel Statement'.split() +
        'I- Question'.split()
    ]


def test_parsing_predictions(dataset, predictions):
    predicted_dataset = predictions_to_dataset(
        original_dataset=dataset,
        predictions=predictions,
        begin_determines_act=False,
        use_joint_coding=False
    )
    assert len(predicted_dataset) == 1
    assert predicted_dataset.call_ids == ['call1']
    call = predicted_dataset.calls[0]
    assert call[0] == FunctionalSegment('hi how are you', dialog_act='Question', speaker='A')
    assert call[1] == FunctionalSegment('long time no see', dialog_act='Statement', speaker='A')
    assert call[2] == FunctionalSegment("i'm", dialog_act='Backchannel', speaker='B')
    assert call[3] == FunctionalSegment("fine", dialog_act='Statement', speaker='B')
    assert call[4] == FunctionalSegment("and you", dialog_act='Question', speaker='B')


def test_parsing_predictions_ignore_continuation_label(dataset, predictions):
    predicted_dataset = predictions_to_dataset(
        original_dataset=dataset,
        predictions=predictions,
        begin_determines_act=True,
        use_joint_coding=False
    )
    assert len(predicted_dataset) == 1
    assert predicted_dataset.call_ids == ['call1']
    call = predicted_dataset.calls[0]
    assert call[0] == FunctionalSegment('hi how are you', dialog_act='Question', speaker='A')
    assert call[1] == FunctionalSegment('long time no see', dialog_act='Statement', speaker='A')
    assert call[2] == FunctionalSegment("i'm fine", dialog_act='Backchannel', speaker='B')
    assert call[3] == FunctionalSegment("and you", dialog_act='Question', speaker='B')


def test_parsing_predictions_joint_coding(dataset, joint_coding_predictions):
    predicted_dataset = predictions_to_dataset(
        original_dataset=dataset,
        predictions=joint_coding_predictions,
        use_joint_coding=True
    )
    assert len(predicted_dataset) == 1
    assert predicted_dataset.call_ids == ['call1']
    call = predicted_dataset.calls[0]
    assert call[0] == FunctionalSegment('hi how are you', dialog_act='Question', speaker='A')
    assert call[1] == FunctionalSegment('long time no see', dialog_act='Statement', speaker='A')
    assert call[2] == FunctionalSegment("i'm", dialog_act='Backchannel', speaker='B')
    assert call[3] == FunctionalSegment("fine", dialog_act='Statement', speaker='B')
    assert call[4] == FunctionalSegment("and you", dialog_act='Question', speaker='B')


@pytest.fixture
def continuation_dataset():
    return DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('hi how are you', dialog_act='Statement', speaker='A'),
            FunctionalSegment("i'm fine", dialog_act='Statement', speaker='B'),
            FunctionalSegment('long time no see', dialog_act='Statement', speaker='A', is_continuation=True),
        ])
    })


@pytest.fixture
def continuation_joint_coding_predictions():
    return [
        'I- I- I- I- O'.split() +
        'I- Statement O'.split() +
        'I- I- I- Statement'.split()
    ]


def test_parsing_predictions_joint_coding_continuation(continuation_dataset, continuation_joint_coding_predictions):
    predicted_dataset = predictions_to_dataset(
        original_dataset=continuation_dataset,
        predictions=continuation_joint_coding_predictions,
        use_joint_coding=True
    )
    assert len(predicted_dataset) == 1
    assert predicted_dataset.call_ids == ['call1']
    call = predicted_dataset.calls[0]
    assert call[0] == FunctionalSegment('hi how are you', dialog_act='Statement', speaker='A')
    assert call[1] == FunctionalSegment("i'm fine", dialog_act='Statement', speaker='B')
    assert call[2] == FunctionalSegment('long time no see', dialog_act='Statement', speaker='A')


@pytest.fixture
def joint_coding_predictions_for_turns():
    return [
        'I- I- I- Question'.split() + 'I- I- I- Statement'.split(),
        'Backchannel Statement'.split() + 'I- Question'.split()
    ]


def test_joint_coding_predictions_to_turns(joint_coding_predictions_for_turns):
    reconstructions = joint_coding_predictions_to_turns(joint_coding_predictions_for_turns)
    assert reconstructions[0][0] == FunctionalSegment('X X X X', 'Question', '')
    assert reconstructions[0][1] == FunctionalSegment('X X X X', 'Statement', '')
    assert reconstructions[1][0] == FunctionalSegment('X', 'Backchannel', '')
    assert reconstructions[1][1] == FunctionalSegment('X', 'Statement', '')
    assert reconstructions[1][2] == FunctionalSegment('X X', 'Question', '')


def test_joint_coding_predictions_to_corpus(joint_coding_predictions_for_turns):
    corpus = joint_coding_predictions_to_corpus(joint_coding_predictions_for_turns)
    assert corpus.calls[0] == Call([
        FunctionalSegment('X X X X', 'Question', ''),
        FunctionalSegment('X X X X', 'Statement', '')
    ])
    assert corpus.calls[1] == Call([
        FunctionalSegment('X', 'Backchannel', ''),
        FunctionalSegment('X', 'Statement', ''),
        FunctionalSegment('X X', 'Question', '')
    ])
