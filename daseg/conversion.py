from itertools import chain
from typing import List, Iterable, Tuple

from cytoolz.itertoolz import sliding_window

from daseg import DialogActCorpus, FunctionalSegment, Call
from daseg.data import NEW_TURN, is_begin_act, is_continued_act, decode_act, BLANK


def joint_coding_predictions_to_corpus(predictions: List[List[str]]) -> DialogActCorpus:
    turns = joint_coding_predictions_to_turns(predictions)
    return turns_to_corpus(turns)


def turns_to_corpus(turns: List[List[FunctionalSegment]]) -> DialogActCorpus:
    calls = (Call(turn) for turn in turns)
    return DialogActCorpus(dialogues={str(i): call for i, call in enumerate(calls)})


def joint_coding_predictions_to_turns(predictions: List[List[str]]) -> List[List[FunctionalSegment]]:
    reconstructions = []
    turn = []
    for turn_prediction in predictions:
        segment = []
        for tag in turn_prediction:
            segment.append(tag)
            if not is_continued_act(tag):
                turn.append(FunctionalSegment(
                    text=' '.join(['X'] * len(segment)),  # preserve the number of words
                    dialog_act=decode_act(segment[-1]),
                    speaker=''
                ))
                segment = []
        if segment:
            turn.append(FunctionalSegment(
                text=' '.join(['X'] * len(segment)),  # preserve the number of words
                dialog_act='?',
                speaker=''
            ))
        reconstructions.append(turn)
        turn = []
    return reconstructions


def predictions_to_dataset(
        original_dataset: DialogActCorpus,
        predictions: List[List[str]],
        begin_determines_act: bool = False,
        use_joint_coding: bool = False
) -> DialogActCorpus:
    dialogues = {}
    for (call_id, call), pred_tags in zip(original_dataset.dialogues.items(), predictions):
        words, _, speakers = call.words_with_metadata(add_turn_token=True)
        assert len(words) == len(pred_tags), \
            f'Mismatched words ({len(words)}) and predicted tags ({len(pred_tags)}) counts'

        def turns(pairs: Iterable[Tuple[str, str, str]]):
            turn = []
            for word, tag, speaker in pairs:
                if NEW_TURN in word or not word:
                    yield turn
                    turn = []
                    continue
                turn.append((word, tag, speaker))
            if turn:
                yield turn

        def segments(turns: Iterable[List[Tuple[str, str, str]]]):
            for turn in turns:
                prev_tag = None
                segment = []
                for word, tag, speaker in turn:
                    if prev_tag is None or tag == prev_tag or (
                            is_begin_act(prev_tag) and is_continued_act(tag) and decode_act(prev_tag) == decode_act(
                        tag)):
                        segment.append((word, tag, speaker))
                    else:
                        yield FunctionalSegment(
                            text=' '.join(w for w, _, _ in segment),
                            dialog_act=decode_act(segment[0][1]),
                            speaker=segment[0][2]
                        )
                        segment = [(word, tag, speaker)]
                    prev_tag = tag
                if segment:
                    yield FunctionalSegment(
                        text=' '.join(w for w, _, _ in segment),
                        dialog_act=decode_act(segment[0][1]),
                        speaker=segment[0][2]
                    )

        def segments_common_continuation_token(turns):
            for turn in turns:
                prev_tag = None
                segment = []
                for word, tag, speaker in turn:
                    if prev_tag is None or is_continued_act(tag):
                        segment.append((word, tag, speaker))
                    else:
                        yield FunctionalSegment(
                            text=' '.join(w for w, _, _ in segment),
                            dialog_act=decode_act(segment[0][1]),
                            speaker=segment[0][2]
                        )
                        segment = [(word, tag, speaker)]
                    prev_tag = tag
                if segment:
                    yield FunctionalSegment(
                        text=' '.join(w for w, _, _ in segment),
                        dialog_act=decode_act(segment[0][1]),
                        speaker=segment[0][2]
                    )

        def segments_joint_coding(turns):

            def inner():
                for turn in turns:
                    segment = []
                    for word, tag, speaker in turn:
                        segment.append((word, tag, speaker))
                        if not is_continued_act(tag):
                            yield FunctionalSegment(
                                text=' '.join(w for w, _, _ in segment),
                                dialog_act=decode_act(segment[-1][1]),
                                speaker=segment[-1][2]
                            )
                            segment = []
                    if segment:
                        yield FunctionalSegment(
                            text=' '.join(w for w, _, _ in segment),
                            dialog_act='?',
                            speaker=segment[0][2]
                        )

            segments = []
            for segment, next_segment in sliding_window(2, chain(inner(), [None])):
                if segment.dialog_act == '?':
                    segment = FunctionalSegment(
                        text=segment.text,
                        dialog_act=next_segment.dialog_act if next_segment is not None else BLANK,
                        speaker=segment.speaker
                    )
                segments.append(segment)
            return segments

        segmentation = segments
        if begin_determines_act:
            segmentation = segments_common_continuation_token
        if use_joint_coding:
            segmentation = segments_joint_coding

        dialogues[call_id] = Call(segmentation(turns(zip(words, pred_tags, speakers))))

    return DialogActCorpus(dialogues=dialogues)
