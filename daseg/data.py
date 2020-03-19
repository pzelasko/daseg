"""
Utilities for retrieving, manipulating and storing the dialog act data.
"""
import random
import re
from itertools import groupby
from typing import NamedTuple, Tuple, List, FrozenSet, Iterable, Dict, AbstractSet

from spacy import displacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.symbols import ORTH
from swda import Transcript, CorpusReader
from tqdm import tqdm

from daseg.resources import DIALOG_ACTS, COLORMAP, get_nlp, get_tokenizer

__all__ = ['FunctionalSegment', 'Call', 'SwdaDataset']


# Symbol used in SWDA to indicate that the dialog act is the same as in previous turn
CONTINUATION = '+'

# Marks beginning of a new tag;
# Without it, we lose the segmentation information when an act is repeated
# and when an act is continued in the next turn
BEGIN_TAG = 'B-'

# Marks this is a continuation of the same act
CONTINUE_TAG = 'I-'

# Blank symbol just for the NEW_TURN symbol
BLANK = 'O'

# Marks new turn for sequence tagging models (e.g. "I am here <TURN> Okay")
NEW_TURN = '<TURN>'


class SwdaDataset:
    def __init__(self, dialogues: Dict[str, 'Call']):
        self.dialogues = dialogues

    @staticmethod
    def from_path(swda_path: str) -> 'SwdaDataset':
        cr = CorpusReader(swda_path)
        dialogues = dict(map(parse_transcript, cr.iter_transcripts()))
        return SwdaDataset(dialogues)

    @staticmethod
    def from_transformers_predictions(preds_file_path: str) -> 'SwdaDataset':
        return read_transformers_preds(preds_file_path)

    @property
    def call_ids(self) -> List[str]:
        return list(self.dialogues.keys())

    @property
    def calls(self) -> List['Call']:
        return list(self.dialogues.values())

    def special_symbols(self) -> FrozenSet[str]:
        """Return the set of symbols in brackets found in SWDA transcripts."""
        uniq_words = {w for segments in self.dialogues.values() for text, _, _, _ in segments for w in text.split()}
        special_symbols = {re.sub(r'[\?\.,!;:]', '', w) for w in uniq_words if w.startswith('<')}
        return frozenset(special_symbols)

    def acts_with_examples(self):
        from cytoolz import groupby
        texts_by_act = groupby(
            lambda tpl: tpl[0],
            sorted(
                (act, text)
                for tr in self.calls
                for text, act, _, _ in tr
            )
        )
        return texts_by_act

    def train_dev_test_split(self) -> Dict[str, 'SwdaDataset']:
        from daseg.splits import train_set_idx, valid_set_idx, test_set_idx
        return {
            'train': self.subset(train_set_idx),
            'dev': self.subset(valid_set_idx),
            'test': self.subset(test_set_idx)
        }

    def dump_for_transformers_ner(self, path: str):
        """Write this dataset to a text file used by Transformers NER recipe."""
        with open(path, 'w') as f:
            for call in tqdm(self.calls):
                lines = to_transformers_ner_dataset(call, special_symbols=self.special_symbols())
                for line in lines:
                    print(line, file=f)
                print(file=f)

    def subset(self, selected_ids: Iterable[str]) -> 'SwdaDataset':
        selected_ids = set(selected_ids)
        dialogues = {call_id: segments for call_id, segments in self.dialogues.items() if call_id in selected_ids}
        return SwdaDataset(dialogues)


class Call(list):
    def render(self, max_turns=None):
        """Render the call as annotated with dialog acts in a Jupyter notebook"""

        # Render DAs options
        nlp = get_nlp()
        rand = random.Random(0)
        colors = COLORMAP[50:]
        rand.shuffle(colors)
        cmap = {k.upper(): col for k, col in zip(DIALOG_ACTS.values(), reversed(colors))}
        displacy_opts = {"colors": cmap}

        # Convert text to Doc
        speakers = {'A:': 'B:', 'B:': 'A:'}
        spk = 'A:'
        for turn_no, (key, group) in enumerate(groupby(self, key=lambda tpl: tpl[2])):
            if max_turns is not None and turn_no > max_turns:
                break
            group = list(group)
            words = ' '.join(t for t, _, _, _ in group).split()
            doc = Doc(nlp.vocab, words=[spk] + words)

            ents = []
            begin = 1
            for (t, act, _, _) in group:
                words = t.split()
                end = begin + len(words)
                ents.append(Span(doc, begin, end, label=act))
                begin = end
            doc.ents = ents

            displacy.render(doc, style="ent", jupyter=True, options=displacy_opts)

            spk = speakers[spk]


class FunctionalSegment(NamedTuple):
    text: str
    dialog_act: str
    speaker: str
    is_continuation: bool = False


def parse_transcript(swda_tr: Transcript) -> Tuple[str, Call]:
    call_id = decode_swda_id(swda_tr)
    brackets = re.compile(r'(<\S+)\s(\S+>)')
    segments = (
        FunctionalSegment(
            text=brackets.sub(
                '\\1\\2',
                ' '.join(utt.text_words(filter_disfluency=True))
            ),
            dialog_act=lookup_or_fix(utt.act_tag),
            speaker=utt.caller
        ) for utt in swda_tr.utterances
    )
    # Resolve '+' into dialog act
    resolved_segments = []
    prev_tag = {'A': 'Other', 'B': 'Other'}  # there seems to be exactly one case where the first DA is '+'
    for text, tag, spk, _ in segments:
        is_continuation = tag == CONTINUATION
        resolved_tag = prev_tag[spk] if is_continuation else tag
        resolved_segments.append(
            FunctionalSegment(
                text=text,
                dialog_act=resolved_tag,
                speaker=spk,
                is_continuation=is_continuation
            )
        )
        prev_tag[spk] = resolved_tag
    return call_id, Call(resolved_segments)


def lookup_or_fix(tag: str) -> str:
    if tag in DIALOG_ACTS:
        return DIALOG_ACTS[tag]
    # https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
    # "We did the clustering by removing the secondary carat-dimensions (^2,^g,^m,^r,^e,^q,^d), with 5 exceptions"
    # (PZ): I added ^t to this list, docs say "about-task",
    #       they used it to indicate which sentences are "on-topic" in SWBD calls
    fixed_tag = re.sub(r'(.+)(\^2|\^g|\^m|\^r|\^e|\^q|\^d|\^t)', '\\1', tag)
    # "we folded the few examples of nn^e into ng, and ny^e into na"
    fixed_tag = 'ng' if tag == 'nn^e' else 'na' if tag == 'ny^e' else fixed_tag
    # (PZ): 'sd(^q)' is very frequent, sv(^q) somewhat frequent;
    #       'sd' is statement-non-opinion, 'sv' statement-opinion,
    #       but upon closer inspection they all seem to be quotes
    fixed_tag = '^q' if any(tag == name for name in ('sd(^q)', 'sv(^q)')) else fixed_tag
    if fixed_tag in DIALOG_ACTS:
        return DIALOG_ACTS[fixed_tag]
    return 'Other'


def decode_swda_id(transcript: Transcript) -> str:
    return f"sw{transcript.swda_filename.split('_')[2].split('.')[0]}"


"""
Transformers IO specific methods.
"""


def to_transformers_ner_dataset(call: List, special_symbols: AbstractSet[str]) -> List[str]:
    """
    Convert a list of functional segments into text representations,
    used by the Transformers library to train NER models.
    """
    tokenizer = get_tokenizer()
    # Avoid spacy tokenizations of the sort <my-token> -> < my - token >
    for sym in special_symbols:
        tokenizer.add_special_case(sym, [{ORTH: sym}])
    lines = []
    prev_spk = None
    prev_tag = {'A': None, 'B': None}
    for utt, tag, who, is_continuation in call:
        tag = '-'.join(tag.split())
        doc = tokenizer(utt)
        tokens = [tok for tok in doc]
        if is_continuation:
            labels = [f'{CONTINUE_TAG}{prev_tag[who]}'] * len(tokens)
        else:
            labels = [f'{BEGIN_TAG}{tag}'] + [f'{CONTINUE_TAG}{tag}'] * (len(tokens) - 1)
        if prev_spk is not None and prev_spk != who:
            tokens = [NEW_TURN] + tokens
            labels = [BLANK] + labels
        for token, label in zip(tokens, labels):
            lines.append(f'{token} {label}')
        prev_spk = who
        prev_tag[who] = tag
    return lines


def is_begin_act(tag):
    return tag.startswith(BEGIN_TAG)


def is_continued_act(tag):
    return tag.startswith(CONTINUE_TAG)


def decode_act(tag):
    if is_begin_act(tag) or is_continued_act(tag):
        return tag[2:]
    return tag


def read_transformers_preds(preds_path: str) -> 'SwdaDataset':
    lines = (l.strip() for l in open(preds_path))

    def calls(lines):
        predictions = []
        for line in lines:
            if not line:
                yield predictions
                predictions = []
                continue
            predictions.append(line)
        if predictions:
            yield predictions


    def turns(call):
        turn = []
        for line in call:
            if NEW_TURN in line or not line:
                yield turn
                turn = []
                continue
            turn.append(line)
        if turn:
            yield turn

    def segments(turn):
        segment = []
        prev_tag = None
        for line in turn:
            text, tag = line.split()
            if prev_tag is None or tag == prev_tag or (is_begin_act(prev_tag) and is_continued_act(tag) and decode_act(prev_tag) == decode_act(tag)):
                segment.append((text, tag))
            else:
                yield segment
                segment = [(text, tag)]
            prev_tag = tag
        if segment:
            yield segment

    resolved_calls = []
    for call in calls(lines):
        resolved_segments = []
        speaker = 'A'
        for turn in turns(call):
            for segment in segments(turn):
                first_tag = segment[0][1]
                resolved_segments.append(
                    FunctionalSegment(
                        text=' '.join(w for w, _ in segment),
                        dialog_act=decode_act(first_tag),
                        speaker=speaker,
                        is_continuation=is_begin_act(first_tag)
                    )
                )
            speaker = {'A': 'B', 'B': 'A'}[speaker]
        resolved_calls.append(Call(resolved_segments))
    # TODO: resolve correct call ids
    return SwdaDataset({str(i): c for i, c in zip(range(1000000), resolved_calls)})


